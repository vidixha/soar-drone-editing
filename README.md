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
  `trace_replication/src/stage1_source_filter.py`: filters a source video
  dataset down to near-static-camera sequences (reusing the ORB-based motion
  probe already validated in `src/motion_probe.py`), and converts them into
  the metadata format ReCamMaster's inference script expects. Currently reads
  GOT-10k's file layout specifically; the actual source dataset is not
  finalized, and this should become configurable rather than assuming one
  dataset's format.
- `trace_replication/src/mavrec_source_filter.py`,
  `trace_replication/src/stage1_build_pairs.py`: a drone-footage POC for
  Stage 1's source corpus, using MAVREC's hovering drone view (semi-static,
  25-45m altitude) instead of GOT-10k. Not a finalized dataset choice, see
  "MAVREC drone-view POC" below.

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
  something it states outright. The training video corpus itself is not
  finalized.

**Training-loop**
- `trace_replication/src/stage1_train.py`: standard training loop over
  precomputed Stage 1 pairs. Optimizer, batch size, and step count are ours.
  The paper only gives those for Stage 2, not Stage 1.
- `trace_replication/src/stage2_lora.py::train_step`: single flow-matching
  training step using the paper's exact stated hyperparameters.
- `trace_replication/requirements.txt`: pip-installable deps, plus notes on
  the four non-PyPI packages (ReCamMaster, CoTracker, DEVA, wan) that need
  their own git installs.

## Data sources

The paper's training data is private for both stages (7,500 static-camera
videos for Stage 1, ~1.1M internal videos for Stage 2), so both stages need a
public substitute. Neither substitute dataset is finalized yet; the source
corpus for Stage 1 and the training corpus for Stage 2 should be configurable
rather than assumed. Stage 2's masks also depend on DEVA, which has not been
run on any candidate dataset yet.

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

The pipeline that chains source-dataset filtering, ReCamMaster re-rendering,
and CoTracker track extraction into the `.pt` training-pair format
`stage1_train.py` expects is not built yet. `stage1_source_filter.py`
currently assumes GOT-10k's specific file layout, since the actual source
dataset is not decided; this will need to become configurable. Each piece can
be run on its own:

```python
# 1. Filter a downloaded source dataset to the near-static-camera subset,
#    then write ReCamMaster's expected metadata.csv. Encoding each qualifying
#    sequence's frames to an mp4 is a separate, ordinary ffmpeg step, not
#    included here.
from stage1_source_filter import filter_got10k, write_metadata_csv
qualifying = filter_got10k(got10k_root=pathlib.Path("/path/to/dataset_root"))
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

### MAVREC drone-view POC

`mavrec_source_filter.py` is a Stage 1 source-corpus POC using MAVREC
(arxiv 2312.04548) instead of GOT-10k. MAVREC's drone camera hovers
(semi-static, 25-45m altitude) rather than flying continuously, which is why
it can satisfy Stage 1's static-camera requirement while still being real
drone footage, unlike VisDrone/UAVDT's continuous-flight clips. This is a POC,
not a finalized dataset choice.

MAVREC is gated on Hugging Face (`huggingface.co/datasets/rjccv/MAVREC`) --
requesting access (name, email, affiliation, country) and acknowledging its
CC-BY license is required before `ACCESS_INSTRUCTIONS.md` reveals the actual
download link, a Google Drive folder -- and we have been through that gate.
Confirmed real layout, from an actual downloaded scene:

```
mavrec_root/
  Annotations/
    aerial_test_aligned_ids.json   # COCO: images[] (file_name, frameID, scene,
                                    # width, height), annotations[] (bbox, category_id)
    ground_test_aligned_ids.json
  unlabelled/
    video_scene_<N>.zip            # ~6.8GB each; contains aerial_scene_<N>.mp4
                                    # (drone view) and ground_scene_<M>.mp4
```

Only the first ~900 frames (30s) of each scene are annotated, and only
sparsely within that window (irregular frame-id gaps, e.g. scene 1 has 146
annotated frames spread across frame 2-897). `mavrec_source_filter.py`
interpolates those sparse boxes into a dense per-frame sequence, which
matches the paper's own description of B_ref as "sparse user-placed key boxes"
interpolated to a dense sequence, rather than assuming dense per-frame
annotations existed. MAVREC is also a multi-object detection dataset (10
categories, no persistent track id), so a single object per scene is picked
via a greedy nearest-center linker (`link_single_object`) across the sparse
annotated frames, since Stage 1 needs one tracked object's box per frame.

`stage1_build_pairs.py` is the data-assembly glue script chaining
`mavrec_source_filter.filter_mavrec` -> ReCamMaster rendering -> CoTracker
tracking into `stage1_train.py`'s `.pt` schema, that `stage1_source_filter.py`
and `stage1_train.py` both flag as not yet built. `render_fn`/`track_fn`/
`feature_fn` are injectable so the assembly and `.pt` schema can be verified
on CPU with stub functions standing in for ReCamMaster and CoTracker (both
GPU-only), before spending GPU time on the real ones:

```bash
python3 stage1_build_pairs.py
```

This has been run twice: once with synthetic data and stub functions (the
`__main__` block, CPU-only), and once against a real downloaded MAVREC scene
(scene 1) end to end -- real sparse COCO boxes interpolated, real 81-frame
window extracted from the real video and confirmed near-static
(translation_norm_median 0.0001, rotation 0.002 degrees, both well under the
static-camera threshold), encoded to an mp4 with `write_clip_mp4`, assembled
into `.pt` pairs with stub ReCamMaster/CoTracker output, and fed into
`stage1_dit.py`'s real flow-matching loss without shape errors. Target-box
values in this POC are still a placeholder (copied from the reference boxes)
since re-localizing the object in ReCamMaster's rendered output needs DEVA,
which has not been run yet (see "Data sources" above). Swapping in the real
GPU calls is passing `recam_wrapper.render_all_trajectories` and
`cotracker_wrapper.extract_point_track_grid` as `render_fn`/`track_fn`
instead of the stubs; no other code change is needed.

### Stage 2

`stage2_data_pipeline.py` builds training pairs from a video corpus (not yet
decided) plus DEVA masks, but does not call DEVA itself; DEVA needs to be run
separately (`demo_automatic.py` in its own repo) to produce the mask
sequences this module consumes. There is no training-loop entrypoint yet, only
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


