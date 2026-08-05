"""
DEVA dry run on CPU: segments the tracked object in scene 1's real (original,
not ReCamMaster-rendered) footage using a box-prompted MobileSAM mask on frame
0, then propagates that identity across all 81 frames using DEVA's own
propagation network (no SAM after frame 0). Compares DEVA's derived per-frame
box (from its output mask) against mavrec_source_filter's interpolated B_ref
box, as a sanity check of DEVA's tracking quality.

This is a dry run, not the real Stage 1 step 4 (re-localizing the object in a
ReCamMaster-rendered clip): ReCamMaster hasn't been run yet (needs GPU), so
there is no rendered clip to re-localize the object in. Running DEVA on the
original clip only tests DEVA's mechanics (segment once, track many frames),
not the actual re-localization task.
"""
import pathlib
import resource
import sys
import time
from argparse import ArgumentParser


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).parent / "Tracking-Anything-with-DEVA"
sys.path.insert(0, str(REPO))
SRC = pathlib.Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from deva.model.network import DEVA
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.object_info import ObjectInfo
from deva.inference.eval_args import add_common_eval_args
from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args
from deva.dataset.utils import im_normalization
from deva.ext.MobileSAM.setup_mobile_sam import setup_model as setup_mobile_sam
from segment_anything import SamPredictor

import mavrec_source_filter as msf


def get_input_frame_for_deva_cpu(image_np: np.ndarray, min_side: int) -> torch.Tensor:
    """Same as demo_utils.get_input_frame_for_deva but without the hardcoded .cuda()."""
    image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255
    image = im_normalization(image)
    if min_side > 0:
        h, w = image_np.shape[:2]
        scale = min_side / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = image.unsqueeze(0)
        image = F.interpolate(image, (new_h, new_w), mode="bilinear", align_corners=False)[0]
    return image


def mask_to_xyxy(mask: np.ndarray) -> tuple | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = ArgumentParser()
    add_common_eval_args(parser)
    add_ext_eval_args(parser)
    add_text_default_args(parser)
    args = parser.parse_args([
        "--model", str(REPO / "saves" / "DEVA-propagation.pth"),
        "--MOBILE_SAM_CHECKPOINT_PATH", str(REPO / "saves" / "mobile_sam.pt"),
        "--size", "480",  # resize shorter side to 480, matching Stage 1's working resolution
    ])
    config = vars(args)
    config["enable_long_term"] = not config["disable_long_term"]
    config["enable_long_term_count_usage"] = (
        config["enable_long_term"]
        and (81 / (config["max_mid_term_frames"] - config["min_mid_term_frames"])
             * config["num_prototypes"]) >= config["max_long_term_elements"]
    )

    print("loading DEVA network on CPU...")
    network = DEVA(config).eval()
    weights = torch.load(config["model"], map_location="cpu")
    network.load_weights(weights)

    print("loading MobileSAM on CPU...")
    sam_checkpoint = torch.load(config["MOBILE_SAM_CHECKPOINT_PATH"], map_location="cpu")
    mobile_sam = setup_mobile_sam()
    mobile_sam.load_state_dict(sam_checkpoint, strict=True)
    mobile_sam.to(device="cpu").eval()
    predictor = SamPredictor(mobile_sam)

    deva = DEVAInferenceCore(network, config=config)
    deva.enabled_long_id()

    mavrec_root = pathlib.Path(
        "/tmp/claude-1002/-home-akshata-projects-soar-drone-editing/"
        "c6ab49e8-ad6e-4229-9c94-eb8ef9e76b1c/scratchpad/mavrec_sample_root"
    )
    video_dir = pathlib.Path(
        "/home/akshata/projects/soar_drone_editing/trace_replication/data/mavrec_raw/extracted"
    )
    qualifying = msf.filter_mavrec(mavrec_root, video_dir, limit=1)
    item = qualifying[0]
    frames_bgr = item["frames_bgr"]
    boxes_norm = item["boxes_normalized"]  # (81, 4) cx,cy,w,h normalized
    h, w = frames_bgr[0].shape[:2]

    def norm_box_to_xyxy(box_norm):
        cx, cy, bw, bh = box_norm
        return (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h

    ref_box_xyxy = norm_box_to_xyxy(boxes_norm[0])
    print(f"frame 0 reference box (xyxy px): {ref_box_xyxy}")

    frame0_rgb = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2RGB)
    predictor.set_image(frame0_rgb)
    if isinstance(predictor.features, tuple):
        # MobileSAM's TinyViT returns (features, None) for SAM-HQ-fork compatibility;
        # vanilla segment_anything's SamPredictor expects a plain tensor.
        predictor.features = predictor.features[0]
    t0 = time.time()
    masks, scores, _ = predictor.predict(box=np.array(ref_box_xyxy), multimask_output=False)
    print(f"SAM box-prompt segmentation on frame 0: {time.time() - t0:.1f}s, score {scores[0]:.3f}")

    image0 = get_input_frame_for_deva_cpu(frame0_rgb, config["size"])
    new_h, new_w = image0.shape[-2:]

    mask0_t = torch.from_numpy(masks[0].astype(np.float32))
    mask0_resized = F.interpolate(mask0_t[None, None], (new_h, new_w), mode="bilinear")[0, 0] > 0.5
    output_mask = torch.zeros((new_h, new_w), dtype=torch.int64)
    output_mask[mask0_resized] = 1
    segments_info = [ObjectInfo(id=1, category_id=None, isthing=True, score=float(scores[0]))]

    def norm_box_to_xyxy_resized(box_norm):
        cx, cy, bw, bh = box_norm
        return (cx - bw / 2) * new_w, (cy - bh / 2) * new_h, (cx + bw / 2) * new_w, (cy + bh / 2) * new_h

    t0 = time.time()
    prob = deva.incorporate_detection(image0, output_mask, segments_info)
    print(f"DEVA incorporate_detection (frame 0): {time.time() - t0:.1f}s")

    ious = []
    empty_frames = 0
    t0 = time.time()
    print(f"RSS after frame 0: {rss_mb():.0f} MB")
    for i in range(1, len(frames_bgr)):
        frame_rgb = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB)
        image = get_input_frame_for_deva_cpu(frame_rgb, config["size"])
        prob = deva.step(image, None, None)
        pred_mask = (prob.argmax(dim=0) == 1).cpu().numpy()
        box = mask_to_xyxy(pred_mask)
        gt_box = norm_box_to_xyxy_resized(boxes_norm[i])
        if box is None:
            empty_frames += 1
            ious.append(0.0)
        else:
            ious.append(iou_xyxy(box, gt_box))
        elapsed = time.time() - t0
        print(f"  frame {i}/80, {elapsed:.1f}s elapsed, {elapsed / i:.2f}s/frame, "
              f"RSS {rss_mb():.0f} MB, running mean IoU so far: {np.mean(ious):.3f}")

    total_time = time.time() - t0
    print(f"\npropagation done: {total_time:.1f}s for 80 frames ({total_time / 80:.2f}s/frame)")
    print(f"empty (lost-track) frames: {empty_frames}/80")
    print(f"mean IoU vs. interpolated B_ref box: {np.mean(ious):.3f}")
    print(f"min IoU: {np.min(ious):.3f}, max IoU: {np.max(ious):.3f}")


if __name__ == "__main__":
    # The real demo scripts (demo_with_text.py etc.) set this globally before running;
    # without it, every forward pass builds an autograd graph that is never freed,
    # causing ~200MB/frame growth and an eventual OOM kill on this 7.7GB-RAM host.
    torch.autograd.set_grad_enabled(False)
    main()
