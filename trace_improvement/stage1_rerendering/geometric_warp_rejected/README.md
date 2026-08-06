# Geometric warp, rejected

For camera trajectories that are a pure rotation with no translation, the new view can be
computed exactly with a classical homography, no model needed. This is geometrically
correct and free to run, but for a hovering drone shot the result reads as a still photo
being panned rather than real video, since there is no parallax. Rejected in favor of
TrajectoryCrafter. Kept here for reference.

## Run

```bash
python geometric_warp.py
```

Expects `original.mp4` in the same folder and the camera extrinsics file used by
ReCamMaster's example data.
