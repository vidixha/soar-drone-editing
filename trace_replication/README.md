# TRACE replication status

TRACE (arXiv 2603.25707) has no released code or checkpoints anywhere: not on
GitHub, not linked from its project page (trace-motion.github.io), not on
Hugging Face. Everything in this directory is our own reimplementation from the
paper's method text, not a port of a reference implementation. Every place we
had to fill a gap the paper doesn't specify is flagged explicitly in the notes/
files and in code comments, so it doesn't get mistaken for a reported detail
later.

## What's built and verified so far

All of the below has been smoke-tested on CPU (shapes, correctness of the box
math, motion scoring on synthetic data). None of it has been run on real data or
real GPU training yet, that's the next phase and needs the two data sources
below in place first.

**Stage 1 (Cross-View Motion Transformation)** — `notes/stage1_spec.md`
- `src/stage1_dit.py`: the 8-layer flow-matching DiT itself, matching the
  paper's stated depth and conditioning (first frame, 25x25 CoTracker point
  grid, reference boxes).
- `src/recam_wrapper.py`: drives ReCamMaster's real public inference script
  across its 10 built-in camera trajectories.
- `src/cotracker_wrapper.py`: drives CoTracker's real public predictor for the
  point-track grid extraction.
- `src/motion_filter.py`, `src/stage1_source_filter.py`: filters GOT-10k down
  to near-static-camera sequences (reusing the ORB-based motion probe already
  validated in `aerial_box_propagation/src/motion_probe.py`), and converts them
  into the metadata format ReCamMaster's inference script expects.

**Stage 2 (Motion-Conditioned Video Resynthesis)** — `notes/stage2_spec.md`
- `src/stage2_boxes_to_masks.py`: renders box sequences into binary spatial
  mask videos.
- `src/stage2_lora.py`: LoRA wrapper for Wan2.1's DiT, the conditioning-injection
  module, and the flow-matching training step, using the exact hyperparameters
  given in the paper (LoRA, 81 frames @ 480x832 @ 24fps, 8k steps, AdamW
  lr=1.2e-5, wd=0.01, batch 32).
- `src/stage2_data_pipeline.py`: builds training pairs from DEVA object masks,
  using a reconstruction-as-training / editing-as-inference construction (train
  on erase-and-replace-at-the-real-location; at inference, swap in a different
  target trajectory from Stage 1's output). This pattern is our design choice,
  reasoned from the paper's stated conditioning set, not something it states
  outright.

**Training-loop harnesses (setup only, not run)**
- `src/stage1_train.py`: standard training loop over precomputed Stage 1 pairs.
  Optimizer/batch-size/step-count are ours -- the paper only gives those for
  Stage 2, not Stage 1.
- `src/stage2_lora.py::train_step`: single flow-matching training step using the
  paper's exact stated hyperparameters.
- `requirements.txt`: pip-installable deps, plus notes on the four non-PyPI
  packages (ReCamMaster, CoTracker, DEVA, wan) that need their own git installs.

Nothing above downloads a dataset or launches a training run. Every training
entrypoint here fails loudly (FileNotFoundError) if pointed at a data_dir with
no precomputed pairs in it, by design, rather than silently fabricating data.

## Data sources chosen (both public, neither is what the paper used)

- **Stage 1 source corpus**: GOT-10k (10k videos, 1.5M+ hand-annotated boxes),
  filtered to the near-static-camera subset, standing in for the paper's 7,500
  static-camera videos. GOT-10k requires manual registration/download from
  http://got-10k.aitestunion.com/, it's not fetchable via a script.
- **Stage 2 training corpus**: OpenVid-1M (huggingface.co/datasets/nkp37/OpenVid-1M,
  ~1M text-video pairs, CC-BY-4.0), run through DEVA for per-object masks since
  it has no object annotations of its own. Closest public scale match to the
  paper's ~1.1M internal videos.

## What's still open / manual

- Both datasets need to actually be downloaded onto whatever machine runs this
  for real; neither is a one-command fetch.
- GOT-10k box re-localization after ReCamMaster re-rendering: ReCamMaster is
  generative, not a geometric warp, so the object needs re-finding in each
  rendered clip. Plan is to reuse DEVA for this too, not yet built.
- "High-quality" filtering criteria for the Stage 1 110k pairs: not specified by
  the paper, not yet decided by us either.
- Box smoothing/noise augmentation parameters and condition-dropping rate for
  Stage 2: paper states these happen, not their values; current defaults in
  `stage2_data_pipeline.py` are reasonable guesses, not reported numbers.
- No actual training run yet, this is all pre-training scaffolding.
