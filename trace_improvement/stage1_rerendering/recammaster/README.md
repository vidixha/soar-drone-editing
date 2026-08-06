# ReCamMaster re-rendering

The first approach tried for re-rendering a drone clip from a new camera angle. Generates
the full new frame with a video diffusion model conditioned on camera pose. Kept here for
reference. On this dataset it invents content in areas the camera reveals for the first
time, for example buildings or sky that do not exist in the real scene. TrajectoryCrafter
does not have this problem and is the approach actually used.

This file also includes the point tracking and object re-localization steps (CoTracker and
DEVA) used to turn a rendered clip into a Stage 1 training pair, and a function to build a
real training pair end to end.

## Run

```bash
modal run pipeline.py::main --step render --cam-type 1
modal run pipeline.py::main --step cotracker --cam-type 1
modal run pipeline.py::main --step deva --cam-type 1 --seed-boxes car
modal run pipeline.py::main --step real_pairs
```
