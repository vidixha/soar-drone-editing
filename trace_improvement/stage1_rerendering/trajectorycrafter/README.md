# TrajectoryCrafter re-rendering

Re-renders a drone clip from a new camera angle. Estimates depth, reprojects the real
scene geometry into the new view, and uses diffusion only to fill in the small gaps left
by that reprojection. This keeps the rest of the frame factually correct instead of
regenerated.

## Run

```bash
modal run pipeline.py::main --step download
modal run pipeline.py::main --step render --theta 10 --phi 0 --r 0 --video-length 49
```

- `theta`, `phi` control rotation in degrees
- `r` controls forward or backward motion
- `video_length` is capped at 49 frames, a limit of the underlying model

Output is saved to a Modal volume and can be pulled locally with
`modal volume get trace-stage1-weights outputs/<label> .`
