# NOTES.md — Aerial Cross-View Box Propagation POC

Everything below was checked against actual output, nothing is assumed. Timestamps of
checks: 2026-07-22.

## Environment

- No NVIDIA GPU on this machine (`nvidia-smi` not found). CPU only. This directly blocks
  **DROID-SLAM**, see below.
- Only Python 3.14 is installed system-wide (no conda, pyenv, uv, docker). Used
  `python3 -m venv .venv` inside `aerial_box_propagation/` for isolation. `fiftyone` 1.19.0
  has prebuilt wheels for 3.14 so no interpreter downgrade was needed.
- OS: Ubuntu 26.04 ("Resolute Raccoon").

## 1. Verification checklist results

### 1.1 Dataset loads

- **FiftyOne path (`fouh.load_from_hub`) does NOT work on this machine.** It requires a
  local MongoDB (`mongod`) instance. The `fiftyone_db` package (which normally bundles a
  platform-specific `mongod` binary) installed cleanly via pip but shipped **no binary at
  all** for this OS, only a stub `__init__.py`. Ubuntu 26.04 is likely too new for
  fiftyone_db's supported binary matrix. Confirmed by direct exception:
  `fiftyone.core.config.FiftyOneConfigError: MongoDB could not be installed on your system`.
- **Plain `datasets.load_dataset` also does not apply.** The HF repo `Voxel51/visdrone-mot`
  contains no parquet files or a dataset loading script, only raw JPEGs plus a FiftyOne
  export (`fiftyone.yml`, `metadata.json`, `samples.json`). It is not structured for the
  generic `datasets` library.
- **Working path used instead: parse `samples.json` directly**, bypassing FiftyOne and
  MongoDB entirely. Downloaded via `huggingface_hub.hf_hub_download`. This file is the
  underlying FiftyOne export (a MongoDB collection dump) and contains everything needed:
  filepath, scene_id, frame_number, per-frame `detections` (list of dicts with
  `bounding_box` in relative `[x, y, w, h]` format, `label`, `index` = persistent track ID,
  `occlusion`, `visibility`, `confidence`), plus scene-level `keypoints` (point tracks,
  not used here). This is the loader implemented in `src/data_loader.py`.

### 1.2 Dataset size and the copy-paste caveat (confirmed as flagged in TASK.md)

The HF dataset card is confirmed to be **copy-pasted from the VisDrone2019-DET card**:
the card's headline text says "This is a FiftyOne version of the VisDrone2019-DET dataset
with **8629 samples**", but the "Dataset Structure" section further down (also
copy-pasted, but from a MOT-specific run) says "Num samples: 2846" and the summary
metadata block (`dataset_summary` in the YAML front matter) says "2846 samples." Actual
count, verified directly by parsing `samples.json`:

- **2846 total frames, across exactly 7 scenes (sequences).** No more, no fewer.
- Per-scene frame counts (all frame-number ranges are contiguous, no gaps):
  - `uav0000086_00000_v`: 464 frames (sporting event, daytime, high pedestrian density)
  - `uav0000182_00000_v`: 363 frames (road, daytime, low pedestrian density)
  - `uav0000117_02622_v`: 349 frames (intersection, night, medium pedestrian density)
  - `uav0000339_00001_v`: 275 frames (intersection, dusk, low pedestrian density)
  - `uav0000137_00458_v`: 233 frames (intersection, daytime, high pedestrian density)
  - `uav0000305_00000_v`: 184 frames (intersection, daytime, low pedestrian density)
  - `uav0000268_05773_v`: 978 frames (road/highway, daytime, low pedestrian density)
- The 2852 image files in the repo (vs 2846 samples) are explained by a `NNNNNNN-K.jpg`
  naming convention: frame 1 of scene 1 is `0000001.jpg`, frame 1 of scenes 2-7 are
  `0000001-2.jpg` through `0000001-7.jpg`, etc. All 2846 referenced filepaths resolve; the
  small file-count vs sample-count discrepancy is just `.gitattributes`/`README.md`/etc.,
  not orphan data.
- This is confirmed to be the **VisDrone-MOT validation split**, not the DET split. Object
  classes present: pedestrian, people, car, van, truck, bus, motor, bicycle, tricycle,
  awning-tricycle, ignored_region.
- Track IDs (`index` field on each detection) are **not globally unique across scenes**,
  only unique within a scene. Any track-selection code must key on `(scene_id, index)`.

### 1.3 Real camera motion per sequence

Measured directly (not assumed) with a coarse ORB-feature + `estimateAffinePartial2D`
proxy (`src/motion_probe.py`), sampling every 8th frame per scene (up to 40 samples) and
computing the per-single-frame-equivalent translation, rotation, and scale change between
consecutive sampled frames. Full numbers in `results/motion_probe.json`. Median per-frame
values:

| scene | translation (px/frame) | rotation (deg/frame) | scale change (/frame) |
|---|---|---|---|
| uav0000086_00000_v | 1.89 | 0.141 | 0.0006 |
| uav0000182_00000_v | 1.59 | 0.005 | 0.0015 |
| uav0000117_02622_v | 2.76 | 0.019 | 0.0006 |
| uav0000339_00001_v | 0.84 | 0.010 | 0.0003 |
| uav0000137_00458_v | 2.30 | 0.008 | 0.0002 |
| uav0000305_00000_v | 1.96 | 0.001 | 0.0015 |
| uav0000268_05773_v | 0.60 | 0.004 | 0.0002 |

**Honest finding: none of the 7 sequences show large per-frame camera motion in this
coarse proxy.** All are sub-3px/frame translation and sub-0.15deg/frame rotation at this
sampling stride. This is plausible for VisDrone (drones mostly loiter/track slowly rather
than doing aggressive maneuvers) but it means:
- There is no sequence here that is a strong analog for the "6-DoF aggressive drift" case
  from the wider project's Module E. Camera motion here is *slow and cumulative*, not
  fast and per-frame-large. Stratification (Section 5) should bucket relative to the
  range actually present in this dataset (low/medium/high within these 7 sequences),
  not against an absolute "aggressive maneuver" bar. This should be stated plainly in
  FINDINGS.md so the "does not invert" / "inverts" conclusion is not overclaimed as
  covering aggressive drift, only the drift regime actually present in VisDrone-MOT.
  Consequences over long horizons (100+ frames) can still be substantial even if
  per-frame motion is small. This is exactly the "drift over time" case Module E part 3
  in the wider project's TASK.md was probing; the same accumulation logic applies here.
- No sequence was found to be perfectly static (0 motion). All 7 have some
  non-zero drone motion, so all are usable, just none are "aggressive."
- `uav0000086_00000_v` has the highest per-frame rotation (0.141 deg/frame), useful as
  the closest thing to a "rotation-dominant" sequence in this set.
- `uav0000268_05773_v` has the lowest translation and rotation, closest to "near-static."

### 1.4 Static objects present and labeled

Confirmed directly. Spot-checked on `uav0000086_00000_v`, frame 1 vs frame 9 (matching the
motion-probe stride): fit a background affine transform from ORB matches, then measured
residual between the transform-predicted position of each track's frame-1 box center and
its actual frame-9 box center. Of 36 tracks present in both frames, **19 had residual under
3px and 26 under 5px**, and the worst residuals (15-38px) were concentrated on pedestrians
clearly walking/running in the footage. This confirms the practical proxy specified in
TASK.md Section 2 ("tracks whose box motion is well explained by a global homography
within a threshold") separates real static objects from real moving ones on this data.

**Threshold chosen and to be used for Protocol A track selection: median per-frame
residual under 3px, using a chained (not single-pair) homography across the full track
lifespan, computed and documented per scene in `src/static_track_selector.py`.** This is
stricter than the 5px figure to reduce false positives (slow movers misclassified as
static). Actual qualifying-track counts per scene will be logged when the selector runs
on full sequences, not just one sampled pair, and reported in `results/`.

### 1.5 Tooling availability

- **Depth-Anything-V3**: no official PyPI package or `transformers` pipeline integration
  yet (that exists for V1/V2 only, e.g. `pipeline(task="depth-estimation",
  model="LiheYoung/depth-anything-small-hf")`). V3 requires cloning
  `ByteDance-Seed/Depth-Anything-3` (or similar) and `pip install -e .`, using the weights
  at `depth-anything/DA3-BASE` on the Hub. It is a plain forward-pass model, so it can run
  on CPU, just slowly. Not yet installed; will install when method 4 is implemented.
- **DROID-SLAM: hard blocker on this machine.** It requires CUDA and a GPU with 11GB+
  memory for both inference and training; there is no CPU-only path in the official repo
  or in any fork found. This machine has no GPU. Per TASK.md's own fallback instruction
  ("If DROID-SLAM is heavy to set up, note it and proceed with Protocol A, which does not
  strictly require it"): **Protocol A (methods 1-3) does not need DROID-SLAM at all.**
  For method 4 (depth-based warping), which does need *some* camera pose, the plan is to
  substitute a lightweight, training-free, CPU-friendly pose estimator built from the same
  ORB feature matches already used for the homography baseline: essential-matrix
  estimation (`cv2.findEssentialMat` + `cv2.recoverPose`) using DA-V3 depth to resolve
  scale ambiguity. This is an explicit substitution for DROID-SLAM, not DROID-SLAM itself,
  and is recorded here as a deviation from the literal task text, permitted by the
  fallback clause. It will be labeled as such in code comments and in FINDINGS.md.
- Wider project's other tools (CoTracker3, SAM3, AMASS, PISCO removal model) are not
  needed for this task per the non-goals section and were not checked.

### 1.6 TRACE Stage 1 protocol, cross-checked against the actual paper

Fetched the arXiv HTML rendering of 2603.25707 directly (the PDF exceeds the fetch size
limit). Confirmed differences from what TASK.md reconstructed from the project page:

- **Baselines confirmed exactly as listed in TASK.md**: linear interpolation,
  MegaSAM-based warping, Depth-Anything-v3-based warping, versus TRACE's learned module.
  This matches the project page's "Cross-View Motion Transformation - Baseline
  Comparison" figure.
- **Metrics differ from TASK.md's assumption.** TRACE's own Stage 1 eval uses only
  **IoU** and **mAP@0.5**, not the richer center-displacement/scale-error/drift-curve
  protocol specified in this task. TASK.md's richer protocol is intentional and stated
  explicitly as more thorough than TRACE's; this is not a contradiction, just worth
  recording so nobody assumes our numbers are directly comparable to TRACE's Table 4
  numbers metric-for-metric. They are comparable only in relative ranking, not in
  absolute metric values.
- **Dataset differs sharply.** TRACE's Stage 1 eval set is **synthetic**: "an evaluation
  set of 100 video pairs using RecamMaster, following the same procedure as our training
  set," with ground truth "derived from the same videos re-rendered with synthetic camera
  paths." It is not evaluated on any real annotated video dataset, aerial or otherwise.
  This strengthens the motivation for this task: TRACE's ground-level result (learned
  module wins, IoU 0.80 / mAP 0.91 on the f2v direction) was measured on rendered
  synthetic camera paths, not on real footage with real annotated tracks. Our result on
  real aerial video with real ground truth is a meaningfully different and arguably more
  informative comparison, not just a domain switch. This point should be made explicitly
  in FINDINGS.md.

## Blocking questions

1. DROID-SLAM cannot run on this machine (see 1.5). Resolved by substitution, not by
   waiting; recorded above and will be called out again in FINDINGS.md limitations.
2. Depth-Anything-V3 is not yet installed (custom repo build required). This blocks
   method 4 (depth-based warping) until done. Not blocking for Protocol A methods 1-3.
3. None of the 7 available sequences have large per-frame camera motion (see 1.3).
   This turned out to be the central limitation of the whole POC, not a minor caveat,
   see FINDINGS.md: VisDrone-MOT's validation split is hovering surveillance footage,
   incapable of testing whether geometric propagation survives real 6-DoF drone motion.
   UAVDT (has explicit camera-view/altitude attributes, more plausible source of real
   translating-camera sequences) is the recommended next dataset.
4. Attempted to use Kaggle's free GPU tier to run DROID-SLAM for real (credentials were
   provided: `~/.kaggle/kaggle.json`, username akshataabhat). Blocked: the API key in
   that file is rejected with `401 Unauthenticated` directly by Kaggle's API (tested via
   raw `curl -u user:key`, not just the CLI, so this is not a client-library issue). The
   newer `kaggle` CLI (2.2.3) has also moved to an OAuth browser-login flow
   (`kaggle auth login`) that needs interactive browser completion and a verification
   code pasted back into the terminal; that flow was started but not completed since it
   requires a human in the loop. Next step: generate a fresh API token at
   kaggle.com/settings > API, or complete the OAuth login interactively.

## Data sources

- Dataset: `Voxel51/visdrone-mot` on Hugging Face, downloaded via `huggingface_hub`.
- TRACE paper: arXiv 2603.25707, HTML rendering via ar5iv, and the project page
  `https://trace-motion.github.io/`.
