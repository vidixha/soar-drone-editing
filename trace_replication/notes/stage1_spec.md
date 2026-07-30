# Stage 1 replication spec (from paper text, no official code exists)

Source: arxiv.org/html/2603.25707, section on Cross-View Motion Transformation Module.
No checkpoints or code were released by the TRACE authors. Everything below is our
own reimplementation from the method description, not a port of their code.

## Model
- 8-layer Diffusion Transformer (DiT), trained with flow matching.
- Inputs:
  - first frame image, encoded via a 3D VAE for visual context
  - a 25x25 grid of sparse point tracks (from CoTracker) capturing camera motion
  - reference bounding boxes from the first frame (dense per-frame boxes, made by
    temporally interpolating sparse user-placed key boxes -> B_ref)
- Output: per-frame bounding box trajectory in the target (moved-camera) view.
- Loss: flow matching, predict velocity between noise and ground truth
  min_theta E[ || (X1 - X0) - v_theta(Xt, t | J_first, P, B_ref) ||^2 ]

## Training data generation
1. Start from 7,500 static-camera videos with bounding boxes on a tracked object.
   (Open question for us: what source corpus. Paper doesn't name it. Needs to be
   near-static camera footage, NOT aerial/drone footage, since ReCamMaster's job is
   to introduce camera motion synthetically.)
2. Re-render each source video with ReCamMaster (github.com/KwaiVGI/ReCamMaster,
   public Wan2.1-based checkpoint: KwaiVGI/ReCamMaster-Wan2.1/step20000.ckpt) using
   its 10 built-in camera trajectories (pan L/R, tilt U/D, zoom in/out, translate
   U/D with rotation, arc L/R with rotation) -> 75,000 rendered clips.
3. Extract a 25x25 grid of point tracks per rendered clip with CoTracker
   (github.com/facebookresearch/co-tracker, cotracker3).
4. Re-derive ground-truth per-frame boxes in each re-rendered clip. ReCamMaster is
   generative, not an exact geometric warp, so the original box cannot just be
   reprojected with a known transform; the object must be re-localized in the
   generated video (paper doesn't specify how; DEVA or a detector re-run on the
   rendered clip is the natural candidate, same tool used for Stage 2 mask
   extraction).
5. Filter to ~110k high-quality pairs (paper's number after filtering; filtering
   criteria not specified beyond "high-quality").

## Eval protocol (for parity checking our replication later)
- 100 video pairs via ReCamMaster, IoU and mAP@0.5 against MegaSAM-warping and
  DepthAnything-v3-warping baselines. Paper reports IoU 0.80 / mAP 0.91 for TRACE
  vs 0.63 / 0.73 for those baselines.

## Open questions to resolve before running this for real
- Source video corpus for the initial 7,500 static clips (not yet decided).
- Exact box re-localization method after ReCamMaster rendering (not specified in
  paper; we plan to reuse DEVA since it's already needed for Stage 2).
- "High-quality" filtering criteria for the 110k pairs (not specified).
