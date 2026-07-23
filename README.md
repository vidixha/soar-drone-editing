# Aerial Cross-View Box Propagation POC

Answers one question: does TRACE's ground-level result (a learned cross-view module
beats geometric baselines) hold on aerial drone video, or does it invert? See
`FINDINGS.md` for the answer and `NOTES.md` for everything verified before building this.

No training, no generative model, no rendering, no edited video output. Output is numbers
and curves only.

## Setup (from a clean checkout)

Only Python 3.14 was available when this was built (no conda/pyenv/uv/docker on the
host); a venv was used for isolation and works fine since `fiftyone` ships 3.14 wheels.
Any Python 3.10+ should work.

```bash
cd aerial_box_propagation
python3 -m venv .venv
source .venv/bin/activate
pip install fiftyone huggingface_hub datasets opencv-python-headless numpy
```

`fiftyone` is imported only for `huggingface_hub`-adjacent tooling during setup; the
actual data path does **not** use FiftyOne's dataset loading, because that requires a
local MongoDB (`mongod`) which had no binary available for this OS (see NOTES.md 1.1).
Ground truth is read directly from the FiftyOne export's `samples.json`.

## Reproducing all results

```bash
cd aerial_box_propagation/src

# 1. Verify camera motion per scene (writes results/motion_probe.json)
python3 motion_probe.py

# 2. Run Protocol A: static-object camera-motion compensation.
#    Downloads and caches VisDrone-MOT frames on first run (data/frames_cache/,
#    ~1-2 GB across all 7 scenes at key-frame stride 5). Writes
#    results/protocol_a_records.json and results/scene_summaries.json.
python3 run_protocol_a.py

# 3. Stratified reporting: motion magnitude, motion type, altitude change,
#    occlusion, horizon, and the drift curve. Writes results/protocol_a_summary.json
#    and prints the same tables reported in FINDINGS.md.
python3 analyze_results.py
```

Everything downloads from Hugging Face (`Voxel51/visdrone-mot`) on first run and is
cached under `data/`; subsequent runs reuse the cache.

## What each module does

- `src/data_loader.py`: parses `samples.json` (the VisDrone-MOT ground truth export)
  directly, bypassing FiftyOne/MongoDB. Frame download/caching via `huggingface_hub`.
- `src/geometry.py`: shared affine-transform and box-math helpers.
- `src/scene_transforms.py`: builds a chained background transform across
  stride-sampled key frames per scene, using ORB features + `estimateAffinePartial2D`.
- `src/static_track_selector.py`: implements TASK.md's practical proxy for "this
  object does not move in the world", using local (not chained-from-far-away) residual
  to avoid conflating the transform chain's own drift with real object motion.
- `src/methods.py`: the four candidate box-propagation methods (method 4, depth-based
  warping, is a stub pending tooling not available on this machine, see NOTES.md 1.5).
- `src/metrics.py`: IoU, center displacement (px and normalized), scale error.
- `src/run_protocol_a.py`: orchestrates the above into per-frame result records.
- `src/analyze_results.py`: stratified aggregation and the drift curve.

## Known gaps

See "Honest limitations" in `FINDINGS.md`. Headline items: only 7 sequences exist in
this dataset (fewer than the 10 the acceptance criteria targeted), method 4 is not
implemented (GPU-dependent tooling unavailable), and Protocol B was not built.
