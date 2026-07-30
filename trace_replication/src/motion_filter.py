"""
Shared camera-motion probe, factored out of the working version in
aerial_box_propagation/src/motion_probe.py (already validated there against real
aerial footage). Used here for the opposite purpose: instead of confirming a
scene HAS real camera motion, we use it to find GOT-10k sequences that DON'T --
the near-static-camera source clips Stage 1 needs before ReCamMaster synthetically
injects motion (see notes/stage1_spec.md, open question 1).

Same ORB + estimateAffinePartial2D approach, so the "static" threshold used here
is directly comparable to the motion numbers already measured on VisDrone/UAVDT.
"""
import numpy as np
import cv2


def estimate_transform(img1_gray: np.ndarray, img2_gray: np.ndarray) -> dict | None:
    orb = cv2.ORB_create(nfeatures=800)
    k1, d1 = orb.detectAndCompute(img1_gray, None)
    k2, d2 = orb.detectAndCompute(img2_gray, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    if len(matches) < 8:
        return None
    matches = sorted(matches, key=lambda m: m.distance)[:200]
    pts1 = np.float32([k1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([k2[m.trainIdx].pt for m in matches])
    M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return None
    dx, dy = M[0, 2], M[1, 2]
    angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    scale = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
    n_inliers = int(inliers.sum()) if inliers is not None else 0
    return dict(dx=float(dx), dy=float(dy), angle_deg=float(angle), scale=float(scale),
                n_matches=len(matches), n_inliers=n_inliers)


def sequence_motion_score(frames_gray: list[np.ndarray], stride: int = 5) -> dict:
    """Median per-frame translation (normalized by image diagonal) and rotation,
    sampled every `stride` frames, matching the aerial motion_probe convention."""
    sampled = frames_gray[::stride]
    transforms = []
    for img1, img2 in zip(sampled, sampled[1:]):
        t = estimate_transform(img1, img2)
        if t is not None:
            transforms.append(t)
    if not transforms:
        return {"valid": False}

    h, w = frames_gray[0].shape
    diag = float(np.hypot(h, w))
    trans_norm = [np.hypot(t["dx"], t["dy"]) / diag for t in transforms]
    rot_deg = [abs(t["angle_deg"]) for t in transforms]
    return {
        "valid": True,
        "n_transforms": len(transforms),
        "translation_norm_median": float(np.median(trans_norm)),
        "rotation_deg_median": float(np.median(rot_deg)),
    }


# A GOT-10k clip qualifies as "static camera" if its median per-sampled-frame
# translation is under this fraction of the image diagonal. Chosen as roughly an
# order of magnitude below the translation_norm values that characterized real
# camera motion in the VisDrone/UAVDT results (those ran from ~0.005 up to
# several percent of the diagonal per frame); not a value from any paper.
STATIC_TRANSLATION_NORM_THRESHOLD = 0.0015
STATIC_ROTATION_DEG_THRESHOLD = 0.3


def is_static_camera(score: dict) -> bool:
    if not score.get("valid"):
        return False
    return (
        score["translation_norm_median"] < STATIC_TRANSLATION_NORM_THRESHOLD
        and score["rotation_deg_median"] < STATIC_ROTATION_DEG_THRESHOLD
    )


if __name__ == "__main__":
    # smoke test: two identical frames (perfectly static) should qualify
    rng = np.random.default_rng(0)
    base = (rng.random((240, 320)) * 255).astype(np.uint8)
    score = sequence_motion_score([base, base, base], stride=1)
    print("identical-frame score:", score, "static:", is_static_camera(score))
