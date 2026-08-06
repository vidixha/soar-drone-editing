# TRACE Stage 1 and Stage 2, drone footage

This is a proof of concept for the TRACE paper applied to the MAVREC drone dataset.
Stage 1 needs a real drone clip re-rendered from a new camera angle, with the tracked
object's box known in both views. Stage 2 edits where that object appears in the video.

## Layout

- `stage1_rerendering/` two approaches to re-rendering a clip from a new camera angle
  - `trajectorycrafter/` the approach that works, depth based reprojection plus diffusion
  - `recammaster/` the first approach tried, kept for reference, hallucinates content
  - `geometric_warp_rejected/` a pure geometry attempt, rejected, kept for reference
- `stage2_editing/` moving or inserting an object in the video, without training a model
  - `classical_compositing/` inpainting and pixel blending
  - `anydoor_insertion/` a pretrained diffusion model for object insertion
- `viewer/` a local page showing every result side by side

## Setup

All GPU pipelines run on [Modal](https://modal.com). Install and authenticate once:

```bash
pip install modal
modal setup
```

Each pipeline downloads its own model weights into a Modal volume the first time it runs.

## Running Stage 1

```bash
modal run stage1_rerendering/trajectorycrafter/pipeline.py::main --step download
modal run stage1_rerendering/trajectorycrafter/pipeline.py::main --step render --theta 10 --phi 0
```

`theta`, `phi`, and `r` control camera rotation and forward motion. See the file's
docstring for details.

## Running Stage 2

Classical approach, no GPU needed beyond LaMa for erasing:

```bash
modal run stage2_editing/classical_compositing/lama_erase.py::main
python stage2_editing/classical_compositing/erase_and_reinsert_v2_poisson.py
```

AnyDoor approach:

```bash
modal run stage2_editing/anydoor_insertion/pipeline.py::main --step download
modal run stage2_editing/anydoor_insertion/pipeline.py::main --step video
```

## Viewing results

```bash
cd viewer
python3 -m http.server 8081 --bind 0.0.0.0
```

Open `http://localhost:8081`.

## Status

Stage 1 re-rendering works well with TrajectoryCrafter across rotation, pan and tilt, and
forward motion. Stage 2 editing is an early proof of concept. Object repositioning with
classical inpainting and blending works. Inserting a new object is harder at this scale
and the AnyDoor approach, while better than plain pixel pasting, is not yet reliable
enough to depend on.
