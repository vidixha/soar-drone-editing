"""Classical homography-based novel-view synthesis for cam_type 1 (pure ~20deg
rotation, zero translation per camera_extrinsics.json). For pure rotation about
the camera center, the mapping from old view to new view is an exact homography
H = K @ R_rel @ inv(K) -- no depth, no neural network, no hallucination for any
pixel that maps from inside the original frame. Only the newly-revealed sliver
(mapped from outside the original frame) is genuinely unknown.

This is a comparison probe against ReCamMaster's diffusion-based render for the
same trajectory, following the same coordinate convention ReCamMaster's own
vis_cam.py uses to interpret camera_extrinsics.json (parse_matrix + get_c2w).
"""
import json
import pathlib

import cv2
import numpy as np

HERE = pathlib.Path(__file__).parent
CAM_JSON = "/tmp/cam_extrinsics.json"
CAM_TYPE = "01"


def parse_matrix(matrix_str):
    # verbatim from ReCamMaster's vis_cam.py
    rows = matrix_str.strip().split('] [')
    matrix = []
    for row in rows:
        row = row.replace('[', '').replace(']', '')
        vals = list(map(float, row.split()))
        if len(vals) == 3:
            matrix.append(vals + [0.])
        else:
            matrix.append(vals)
    return np.array(matrix)


def get_c2w(w2cs, transform_matrix, relative_c2w=True):
    if relative_c2w:
        target_cam_c2w = np.eye(4)
        abs2rel = target_cam_c2w @ w2cs[0]
        ret_poses = [target_cam_c2w] + [abs2rel @ np.linalg.inv(w2c) for w2c in w2cs[1:]]
    else:
        ret_poses = [np.linalg.inv(w2c) for w2c in w2cs]
    ret_poses = [transform_matrix @ x for x in ret_poses]
    return np.array(ret_poses, dtype=np.float32)


def load_c2ws(cam_type, n_frames):
    data = json.load(open(CAM_JSON))
    mats = [parse_matrix(data[f"frame{i}"][f"cam{cam_type}"]) for i in range(n_frames)]
    cameras = np.transpose(np.stack(mats), (0, 2, 1))
    w2cs = []
    for cam in cameras:
        if cam.shape[0] == 3:
            cam = np.vstack((cam, np.array([[0, 0, 0, 1]])))
        cam = cam[:, [1, 2, 0, 3]]
        cam[:3, 1] *= -1.
        w2cs.append(np.linalg.inv(cam))
    transform_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    return get_c2w(w2cs, transform_matrix, True)  # (n_frames, 4, 4)


cap = cv2.VideoCapture(str(HERE / "original.mp4"))
frames = []
while True:
    ret, f = cap.read()
    if not ret:
        break
    frames.append(f)
cap.release()
n = len(frames)
h, w = frames[0].shape[:2]

c2ws = load_c2ws(CAM_TYPE, n)
R0 = c2ws[0][:3, :3]

# Assume a plausible horizontal FOV for this drone footage (no intrinsics are
# published with camera_extrinsics.json -- ReCamMaster's diffusion model learns
# this implicitly from training data; here we need an explicit value).
FOV_DEG = 60.0
focal_px = (w / 2) / np.tan(np.radians(FOV_DEG / 2))
K = np.array([[focal_px, 0, w / 2], [0, focal_px, h / 2], [0, 0, 1]])
K_inv = np.linalg.inv(K)

out_frames = [frames[0]]
for t in range(1, n):
    Rt = c2ws[t][:3, :3]
    R_rel = R0.T @ Rt  # rotation from the (assumed-fixed) original camera pose to frame t's target pose
    H = K @ R_rel @ K_inv
    # Warp frame t's ACTUAL content (preserves real object motion, e.g. moving
    # cars), not frame0 repeated -- reusing frame0 for every t was a bug that
    # made the whole clip look like one static photo being panned.
    warped = cv2.warpPerspective(frames[t], H, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    out_frames.append(warped)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
tmp_path = HERE / "geometric_warp_tmp.mp4"
writer = cv2.VideoWriter(str(tmp_path), fourcc, 30.0, (w, h))
for f in out_frames:
    writer.write(f)
writer.release()
print(f"wrote {tmp_path}, {n} frames, assumed FOV={FOV_DEG}deg, focal_px={focal_px:.1f}")
