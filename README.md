# TRACE replication status

Most of the below has been tested on CPU - shapes, box math correctness, and
motion scoring on synthetic data. I ran it end to end on a GPU, using the real pretrained Wan2.1 DiT
class, our LoRA wrapper, and our conditioning-injection module, on synthetic data. Forward and backward passes both completed without error. This confirms the module shapes and the training step's API
calls are lined up with the real Wan2.1 architecture. 


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

## Data sources chosen 
- **Stage 1 source corpus**: GOT-10k (10k videos, 1.5M+ hand-annotated boxes),
  filtered to the near-static-camera subset, used in place of the paper's
  7,500 static-camera videos, which are not public. GOT-10k requires manual
  registration and download from http://got-10k.aitestunion.com/. It is not
  fetchable via a script.
- **Stage 2 training corpus**: OpenVid-1M (huggingface.co/datasets/nkp37/OpenVid-1M,
  ~1M text-video pairs, CC-BY-4.0), used in place of the paper's ~1.1M
  internal videos, which are not public. OpenVid-1M is the closest public
  dataset at matching scale. It has no object annotations of its own, so
  per-object masks need to come from DEVA. TBD: DEVA has not been run on
  OpenVid-1M yet. `stage2_data_pipeline.py` consumes DEVA's output format;
  it does not call DEVA itself.

## How to run

```bash
cd trace_replication
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Non-PyPI deps (ReCamMaster, CoTracker, DEVA, wan), see requirements.txt for
# each one's install command.
```

### Stage 1

The pipeline that chains GOT-10k filtering, ReCamMaster re-rendering, and
CoTracker track extraction into the `.pt` training-pair format
`stage1_train.py` expects is not built yet. Each piece can be run on its own:

```python
# 1. Filter a downloaded GOT-10k directory to the near-static-camera subset,
#    then write ReCamMaster's expected metadata.csv. Encoding each qualifying
#    sequence's frames to an mp4 is a separate, ordinary ffmpeg step, not
#    included here.
from stage1_source_filter import filter_got10k, write_metadata_csv
qualifying = filter_got10k(got10k_root=pathlib.Path("/path/to/got10k_root"))
write_metadata_csv(qualifying, video_dir=pathlib.Path("data/stage1_videos"),
                    out_csv=pathlib.Path("data/stage1_source/metadata.csv"))

# 2. Re-render the filtered clips across ReCamMaster's 10 camera trajectories.
from recam_wrapper import render_all_trajectories
render_all_trajectories(
    recammaster_repo=pathlib.Path("/path/to/ReCamMaster"),
    dataset_path=pathlib.Path("data/stage1_source"),
    output_dir=pathlib.Path("data/stage1_rendered"),
    ckpt_path=pathlib.Path("/path/to/step20000.ckpt"),
)

# 3. Extract point-track grids from each rendered clip with CoTracker.
from cotracker_wrapper import extract_point_track_grid
tracks = extract_point_track_grid(video)  # video: (1, T, 3, H, W), see docstring
```

Once `.pt` pairs matching `stage1_train.py`'s expected schema exist under a
data directory, training runs with:

```bash
python3 stage1_train.py --data_dir data/stage1_pairs --out_dir checkpoints/stage1
```

`--num_steps`, `--batch_size`, `--lr`, and `--device` are also available; see
`stage1_train.py --help`. These defaults are ours, not paper-reported values,
since the paper only gives Stage 1's architecture, not its training recipe.

### Stage 2

`stage2_data_pipeline.py` builds training pairs from OpenVid-1M clips and
DEVA masks, but does not call DEVA itself; DEVA needs to be run separately
(`demo_automatic.py` in its own repo) to produce the mask sequences this
module consumes. There is no training-loop entrypoint yet, only
`stage2_lora.py::train_step`, a single flow-matching step meant to be called
from a training loop once the data pipeline is wired up:

```python
from stage2_lora import build_lora_wan_dit, ConditionInjector, train_step

wan_pipe.dit = build_lora_wan_dit(wan_pipe.dit)
injector = ConditionInjector(latent_channels=16, cond_channels=18)
optimizer = torch.optim.AdamW(wan_pipe.dit.parameters(), lr=1.2e-5, weight_decay=0.01)

for batch in dataloader:  # dataloader not built yet, see stage2_data_pipeline.py
    loss = train_step(wan_pipe, injector, batch, optimizer)
```

Individual modules can be run standalone as a shape check without any real
data or model weights:

```bash
python3 stage2_boxes_to_masks.py
python3 stage2_lora.py
```


