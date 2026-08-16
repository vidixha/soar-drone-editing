# Hybrid pipeline: NL router + removal + trajectory

A natural-language-driven aerial video editor. An instruction is parsed into
an ordered list of edit operations, scheduled so content/geometry edits run
before appearance edits, then dispatched to either a local training-free
module or a real GPU render on Modal.

This branch ships three pieces: the **router**, **object removal**, and
**trajectory (camera viewpoint) editing**. Weather modules and the object
insertion prototype live on a separate branch.

## What's included

- `hybrid_pipeline/router.py` -- parses NL instructions into an op-list
  (`remove`, `insert`, `trajectory`), schedules them in canonical order, and
  dispatches each to its module. Rule-based parser; an LLM backend is a
  documented drop-in (`parse_llm`, not wired in) for open-vocabulary
  instructions -- it would only ever produce the op-list, never touch pixels.
- `hybrid_pipeline/modules.py` -- **object removal**: classical
  background-reveal. Builds one sharp static plate from real pixels across
  the clip (drift-aligned, not a blurry average) and blends it in wherever
  the target object was. No model, no training. Assumes a near-static
  (hover) camera -- the case it was built and validated for.
- `hybrid_pipeline/trajcrafter_pipeline.py` -- **trajectory editing**: a
  Modal app wrapping [TrajectoryCrafter](https://github.com/TrajectoryCrafter/TrajectoryCrafter)
  (depth-based reprojection + diffusion gap-fill) for real camera viewpoint
  changes. Zero-shot (released checkpoints, no training by us).
- `hybrid_pipeline/depth_extract.py` -- Depth Anything V2 on Modal; single-frame
  depth snapshot, used as the default post-trajectory depth refresh.
- `hybrid_pipeline/trace_anything_depth.py` -- [Trace Anything](https://github.com/ByteDance-Seed/TraceAnything)
  on Modal; genuinely per-frame, temporally-consistent depth for a whole
  clip in one feed-forward pass. Optional, better alternative to the
  Depth Anything V2 refresh (`--use-trace-anything`). CC-BY-NC-4.0 weights.

## Why depth gets refreshed after trajectory

Once the camera viewpoint changes, the old depth map no longer matches the
scene. Feeding stale depth into anything downstream of it produces visible
bugs (confirmed while building this: mismatched depth broke a since-removed
weather module's collision physics after a trajectory edit -- pixels
"landed" in the wrong place because the geometry they were checked against no
longer described the current frame). `router.py` re-fetches depth for the
new viewpoint every time trajectory runs, via either Depth Anything V2
(default) or Trace Anything (`--use-trace-anything`, more correct, costs an
extra GPU call).

## Setup

Everything GPU-side runs on [Modal](https://modal.com). You need:

```bash
pip install modal
modal setup   # authenticate
```

Deploy the two Modal apps trajectory depends on:

```bash
modal deploy hybrid_pipeline/trajcrafter_pipeline.py
modal deploy hybrid_pipeline/depth_extract.py
# optional, for --use-trace-anything:
modal deploy hybrid_pipeline/trace_anything_depth.py
```

`trajcrafter_pipeline.py`'s weight download (~98GB: TrajectoryCrafter,
DepthCrafter, SVD, CogVideoX-Fun-V1.1-5B, BLIP2) is CPU-billed and one-time:

```bash
modal run hybrid_pipeline/trajcrafter_pipeline.py --step download
```

`trace_anything_depth.py` needs its own checkpoint (~2.6GB) on the same
Modal volume/workspace before first use.

Local deps (removal runs entirely on your machine, no GPU):

```bash
pip install opencv-python-headless numpy
# ffmpeg must be on PATH
```

## Running it

Removal only, local, no GPU, no cost:

```bash
python hybrid_pipeline/router.py --clip your_clip.mp4 \
    --instruction "remove the cars" --out out.mp4
```

Removal + a real trajectory render (fires a paid GPU call on Modal):

```bash
python hybrid_pipeline/router.py --clip your_clip.mp4 \
    --instruction "remove the cars and orbit left" \
    --out out.mp4 --trajectory-gpu
```

Add `--use-trace-anything` to refresh depth after trajectory with Trace
Anything instead of the single-frame default. Add `--reuse-trajectory-cache`
to skip a real re-render and reuse the last cached post-trajectory result
(`/tmp/soar_router_traj_cache.*`) -- useful when iterating on anything
downstream of trajectory without re-paying for a ~5-6 minute A100 render
each time.

## Known limitations

- **Removal** assumes a near-static camera (hover shot). It's a good
  assumption for the aerial footage this was built against, not a general
  solver for arbitrary camera motion.
- **Trajectory output is softer than the source footage.** TrajectoryCrafter's
  diffusion backbone (CogVideoX-Fun-V1.1-5B) visibly softens fine detail in
  regions it has to synthesize rather than warp. Passing an explicit
  `--sample_size` to sharpen this reliably crashes the model's own gradual-mode
  compositing (`RuntimeError: Sizes of tensors must match ... Expected size
  480 but got size 384`) -- confirmed at two different resolutions, not a
  tuning problem. `router.py` deliberately leaves it unset.
- **Insertion** is not implemented on this branch (parsed and scheduled, but
  a no-op at dispatch).
