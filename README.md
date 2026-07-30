# TRACE replication status

## What's built and verified so far

Most of the below has been tested on CPU. Shapes, box math
correctness, motion scoring on synthetic data. The Stage 2 training step is the
exception. We ran it end to end on a GPU, using the real pretrained Wan2.1 DiT
class, our LoRA wrapper, and our conditioning-injection module, on synthetic
tensors instead of real data. Forward and backward passes both completed
without error. This confirms the module shapes and the training step's API
calls line up with the real Wan2.1 architecture. It does not confirm the model
learns anything useful; that requires real data and a real training run.


**Stage 1 (Cross-View Motion Transformation).** See
`trace_replication/notes/stage1_spec.md`.
- `trace_replication/src/stage1_dit.py`: the 8-layer flow-matching DiT itself,
  matching the paper's stated depth and conditioning (first frame, 25x25
  CoTracker point grid, reference boxes).
- `trace_replication/src/recam_wrapper.py`: drives ReCamMaster's real public
  inference script across its 10 built-in camera trajectories.
- `trace_replication/src/cotracker_wrapper.py`: drives CoTracker's real public
  predictor for the point-track grid extraction.
- `trace_replication/src/motion_filter.py`,
  `trace_replication/src/stage1_source_filter.py`: filters GOT-10k down to
  near-static-camera sequences (reusing the ORB-based motion probe already
  validated in `src/motion_probe.py`), and converts them into the metadata
  format ReCamMaster's inference script expects.

**Stage 2 (Motion-Conditioned Video Resynthesis).** See
`trace_replication/notes/stage2_spec.md`.
- `trace_replication/src/stage2_boxes_to_masks.py`: renders box sequences into
  binary spatial mask videos.
- `trace_replication/src/stage2_lora.py`: LoRA wrapper for Wan2.1's DiT, the
  conditioning-injection module, and the flow-matching training step, using
  the exact hyperparameters given in the paper (LoRA, 81 frames @ 480x832 @
  24fps, 8k steps, AdamW lr=1.2e-5, wd=0.01, batch 32).
- `trace_replication/src/stage2_data_pipeline.py`: builds training pairs from
  DEVA object masks, using a reconstruction-as-training / editing-as-inference
  construction (train on erase-and-replace-at-the-real-location; at inference,
  swap in a different target trajectory from Stage 1's output). This pattern
  is our design choice, reasoned from the paper's stated conditioning set, not
  something it states outright.

**Training-loop**
- `trace_replication/src/stage1_train.py`: standard training loop over
  precomputed Stage 1 pairs. Optimizer, batch size, and step count are ours.
  The paper only gives those for Stage 2, not Stage 1.
- `trace_replication/src/stage2_lora.py::train_step`: single flow-matching
  training step using the paper's exact stated hyperparameters.
- `trace_replication/requirements.txt`: pip-installable deps, plus notes on
  the four non-PyPI packages (ReCamMaster, CoTracker, DEVA, wan) that need
  their own git installs.

Nothing above downloads a dataset or launches a training run. Every training
entrypoint here fails loudly (FileNotFoundError) if pointed at a data_dir with
no precomputed pairs in it, by design, rather than silently fabricating data.

## Data sources chosen (both public, neither is what the paper used)

- **Stage 1 source corpus**: GOT-10k (10k videos, 1.5M+ hand-annotated boxes),
  filtered to the near-static-camera subset, standing in for the paper's 7,500
  static-camera videos. GOT-10k requires manual registration and download from
  http://got-10k.aitestunion.com/. It is not fetchable via a script.
- **Stage 2 training corpus**: OpenVid-1M (huggingface.co/datasets/nkp37/OpenVid-1M,
  ~1M text-video pairs, CC-BY-4.0), run through DEVA for per-object masks since
  it has no object annotations of its own. Closest public scale match to the
  paper's ~1.1M internal videos.


