"""
Thin orchestration wrapper around ReCamMaster (github.com/KwaiVGI/ReCamMaster), used
to re-render static-camera source clips with each of its 10 built-in synthetic
camera trajectories. This is step 2 of the Stage 1 data generation recipe (see
notes/stage1_spec.md) -- it does not reimplement ReCamMaster, it drives its public
inference script.

Requires, on the machine actually running this (not this dev machine):
  git clone https://github.com/KwaiVGI/ReCamMaster.git
  cd ReCamMaster && pip install -e .
  python download_wan2.1.py
  (download KlingTeam/ReCamMaster-Wan2.1/step20000.ckpt from HF into
   models/ReCamMaster/checkpoints/ -- confirmed public, not gated)

ReCamMaster's inference script expects a dataset dir with videos/ and a
metadata.csv (file_name, text), 81-frame clips at 480x832. cam_type is a string
"1".."10" selecting one of the 10 built-in trajectories (see CAM_TYPES below).
"""
import csv
import pathlib
import subprocess

CAM_TYPES = {
    "1": "pan_right",
    "2": "pan_left",
    "3": "tilt_up",
    "4": "tilt_down",
    "5": "zoom_in",
    "6": "zoom_out",
    "7": "translate_up_rot",
    "8": "translate_down_rot",
    "9": "arc_left_rot",
    "10": "arc_right_rot",
}


def build_metadata_csv(video_dir: pathlib.Path, captions: dict[str, str], out_csv: pathlib.Path) -> None:
    """captions maps video file_name -> text prompt (needed by ReCamMaster's T2V backbone)."""
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "text"])
        for file_name, text in captions.items():
            assert (video_dir / file_name).exists(), f"missing {file_name}"
            writer.writerow([file_name, text])


def render_all_trajectories(
    recammaster_repo: pathlib.Path,
    dataset_path: pathlib.Path,
    output_dir: pathlib.Path,
    ckpt_path: pathlib.Path,
    cfg_scale: float = 5.0,
) -> list[pathlib.Path]:
    """
    Runs inference_recammaster.py once per cam_type (1..10) so every source clip in
    dataset_path/metadata.csv gets re-rendered along all 10 trajectories, matching
    the paper's "10 different dynamic camera paths" per source video.

    Returns the list of per-cam_type output directories actually produced.
    """
    produced = []
    for cam_type in CAM_TYPES:
        cam_out = output_dir / f"cam_type{cam_type}"
        cmd = [
            "python",
            str(recammaster_repo / "inference_recammaster.py"),
            "--dataset_path", str(dataset_path),
            "--ckpt_path", str(ckpt_path),
            "--output_dir", str(output_dir),
            "--cam_type", cam_type,
            "--cfg_scale", str(cfg_scale),
        ]
        subprocess.run(cmd, check=True, cwd=recammaster_repo)
        produced.append(cam_out)
    return produced
