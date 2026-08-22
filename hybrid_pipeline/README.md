# AERIE (Aerial Editing via Routed Instructions & Execution)

A natural-language-driven aerial video editor. An instruction is parsed into
an ordered list of edit operations, scheduled so content/geometry edits run
before appearance edits, then dispatched to either a local training-free
module or a real GPU render via a pluggable backend.

This branch ships four pieces: the **router**, **object removal**,
**trajectory (camera viewpoint) editing**, and **weather**. The object
insertion prototype is still a no-op at dispatch.

## What's included

- `hybrid_pipeline/router.py` -- parses NL instructions into an op-list
  (`remove`, `insert`, `trajectory`, `weather`), schedules them in canonical order, and
  dispatches each to its module. Two parsers, chosen via `--parser`: regex
  (default, free, instant, fixed keyword list) or a small local LLM
  (`--parser llm`). Never imports a GPU provider directly -- all GPU work
  goes through `gpu_backend.GPUBackend`.
- `hybrid_pipeline/llm_parser.py` -- the LLM parser: **Qwen2.5-0.5B-Instruct**
  (494M params), CPU-only, ~1.6GB RAM, ~15-25s/call. Deliberately small --
  this is structured extraction from one short sentence, not open-ended
  reasoning, so a frontier model would be real cost for no accuracy gain.
  Generalizes past the regex parser's fixed keyword list: e.g. "get the
  vehicles out of the frame please" (zero keyword overlap with the regex
  vocabulary) correctly parses to `remove(vehicles)`; the regex parser
  returns `[]` on the same input by construction. Real finding while
  building this: a *longer, more thorough* system prompt measurably
  *degraded* this model's structured-output reliability (see the module
  docstring) -- small models need minimal prompts, unlike frontier ones.
- `hybrid_pipeline/modules.py` -- **object removal**: classical
  background-reveal. Builds one sharp static plate from real pixels across
  the clip (drift-aligned, not a blurry average) and blends it in wherever
  the target object was. No model, no training. Assumes a near-static
  (hover) camera -- the case it was built and validated for.
- `hybrid_pipeline/weather.py` -- **weather** dispatch. All four kinds use
  `reconstruction_weather/` (metric cameras + AerialMetric depth + 3D
  particles). Sandstorm is the snow flake engine with a sand colour and
  more wind. Fog is metric-depth haze plus sparse motes. `--weather-gpu`
  sends reconstruct+simulate through `weather_pipeline.py` on Modal.
- `hybrid_pipeline/weather_pipeline.py` -- Modal app (`aerie-weather`) that
  runs Pi3X + AerialMetric + particle sim on an L4.
- `hybrid_pipeline/gpu_backend.py` -- the abstract interface (`GPUBackend`:
  `render_trajectory`, `depth_sequence`, `depth_single_frame`, `apply_weather`)
  and a small factory (`get_backend(name)`). Swap providers by implementing
  this interface and registering it -- nothing else changes.
- `hybrid_pipeline/modal_backend.py` -- the one shipped implementation,
  backed by [Modal](https://modal.com). Default (`--backend modal`).
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

`trajcrafter_pipeline.py`, `depth_extract.py`, `trace_anything_depth.py`,
and `weather_pipeline.py` are Modal *server-side* app definitions -- what
actually runs on the GPU. `modal_backend.py` is the *client-side* adapter
that calls into them. Only `modal_backend.py` needs to change (or be
replaced) to run this router against a different GPU provider; the pipeline
files above stay as one reference implementation of what a backend needs to
expose.

## Why depth gets refreshed after trajectory

Once the camera viewpoint changes, the old depth map no longer matches the
scene. Feeding stale depth into anything downstream of it produces visible
bugs (confirmed while building this: mismatched depth broke a since-removed
weather module's collision physics after a trajectory edit -- pixels
"landed" in the wrong place because the geometry they were checked against no
longer described the current frame). `router.py` re-fetches depth for the
new viewpoint every time trajectory runs, via either Depth Anything V2
(default) or Trace Anything (`--use-trace-anything`, more correct, costs an
extra GPU call). Metric weather then reconstructs from those new frames
before simulating.

## Setup

Everything GPU-side runs on [Modal](https://modal.com). You need:

```bash
pip install modal
modal setup   # authenticate
```

Deploy the Modal apps trajectory and weather depend on:

```bash
modal deploy hybrid_pipeline/trajcrafter_pipeline.py
modal deploy hybrid_pipeline/depth_extract.py
# optional, for --use-trace-anything:
modal deploy hybrid_pipeline/trace_anything_depth.py
# weather on Modal (Pi3X + AerialMetric + particle sim):
git submodule update --init third_party/Pi3 third_party/AerialMetric
modal deploy hybrid_pipeline/weather_pipeline.py
```

`trajcrafter_pipeline.py`'s weight download (~98GB: TrajectoryCrafter,
DepthCrafter, SVD, CogVideoX-Fun-V1.1-5B, BLIP2) is CPU-billed and one-time:

```bash
modal run hybrid_pipeline/trajcrafter_pipeline.py --step download
```

`trace_anything_depth.py` needs its own checkpoint (~2.6GB) on the same
Modal volume/workspace before first use.

`weather_pipeline.py` needs AerialMetric + Pi3X weights on its Modal
volume (CPU-billed, one-time):

```bash
modal run hybrid_pipeline/weather_pipeline.py --step download
```

Local deps (removal runs entirely on your machine, no GPU):

```bash
pip install opencv-python-headless numpy
# ffmpeg must be on PATH
```

Weather on Modal uses `--weather-gpu` (same GPU backend as trajectory).
Optional local GPU reconstruction env (Pi3X + AerialMetric) if you skip
that flag:

```bash
git submodule update --init third_party/Pi3 third_party/AerialMetric
bash hybrid_pipeline/setup_weather.sh --download-checkpoint
cp hybrid_pipeline/.env.example hybrid_pipeline/.env
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

Weather uses the same `--instruction` surface. `--weather-gpu` reconstructs
and simulates on Modal (L4). Without it, local Pi3X + AerialMetric is used
(or `--reconstruction-dir` if you already have one):

```bash
python hybrid_pipeline/router.py --clip your_clip.mp4 \
    --instruction "add snow" --out out.mp4 --weather-gpu

python hybrid_pipeline/router.py --clip your_clip.mp4 \
    --instruction "remove the cars and add heavy snow" \
    --out out.mp4 --weather-gpu

python hybrid_pipeline/router.py --clip your_clip.mp4 \
    --instruction "make it rainy" --out out.mp4 --weather-gpu

python hybrid_pipeline/router.py --clip your_clip.mp4 \
    --instruction "add fog" --out out.mp4 --weather-gpu

python hybrid_pipeline/router.py --clip your_clip.mp4 \
    --instruction "add a sandstorm" --out out.mp4 --weather-gpu
```

Phrases the regex parser already knows: `snow` / `snowy` / `blizzard`,
`rain` / `rainy` / `drizzle`, `fog` / `foggy` / `haze`, `sandstorm`,
plus `light` / `medium` / `heavy`.

Add `--use-trace-anything` to refresh depth after trajectory with Trace
Anything instead of the single-frame default. Add `--reuse-trajectory-cache`
to skip a real re-render and reuse the last cached post-trajectory result
(`/tmp/soar_router_traj_cache.*`) -- useful when iterating on anything
downstream of trajectory without re-paying for a ~5-6 minute A100 render
each time. `--backend modal` is the default and only shipped option today;
see `gpu_backend.py` to add another.

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
- **Weather** collides with the reconstructed static depth cloud (snow and
  sandstorm flakes; rain streaks). Fog is metric-depth haze, not a 2D overlay.
