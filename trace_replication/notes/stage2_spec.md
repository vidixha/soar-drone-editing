# Stage 2 replication spec (from paper text, no official code exists)

Source: user-provided paragraph (arxiv 2603.25707, Model architecture and training).
Same situation as Stage 1: no checkpoint or code released, reimplementing from the
method description only.

## Model
- Base: Wan2.1 1.4B (matches the "1.3B" T2V backbone used elsewhere in this
  ecosystem, e.g. ReCamMaster and VACE-Wan2.1-1.3B-Preview -- the paper rounds it
  differently but it is the same family of model we already have working
  experience with from the VACE side of this project).
- Adaptation method: LoRA fine-tuning (not full fine-tune).
- Conditioning set C:
  - text prompt T_text
  - first frame J_first
  - masked video V_mask (source video with the object region removed)
  - synthesis boxes M_obj (dense per-frame boxes = where the object should appear
    in the OUTPUT, i.e. the new/edited trajectory)
  - inpainting boxes M_inpaint (dense per-frame boxes = where the object WAS in
    the source, i.e. the region to erase/inpaint)
- Loss: flow matching, same form as Stage 1:
  min_Phi E[ || (X1 - X0) - v_Phi(Xt, t | C) ||^2 ]
- v_Phi iteratively integrated from pure noise to a clean video latent, decoded
  with a 3D VAE.

## Training hyperparameters (given explicitly)
- 81-frame clips, 480x832 resolution, 24 fps
- 8,000 training steps
- AdamW, lr 1.2e-5, weight decay 0.01
- batch size 32

## Data (from earlier extraction, not in the pasted paragraph)
- ~1.1M videos from an internal dataset (80% long-shot), not available to us --
  needs a public substitute corpus.
- Objects extracted via DEVA tracker.
- Augmentation: box smoothing + noise, random condition dropping.

## Implementation approach
Two of the five conditioning signals (M_obj, M_inpaint) are box sequences, not
pixel content -- they need to be rendered into spatial maps before they can be
fed to a video DiT. The most direct approach, and the one closest to what VACE
already does for its own mask conditioning (which we have running code for in
aerial_box_propagation/modal/vace_modal.py), is:
  1. Render M_obj and M_inpaint as binary mask videos, one frame per box, same
     resolution as the main video.
  2. VAE-encode: masked video V_mask, M_obj mask video, M_inpaint mask video.
  3. Concatenate their latents channel-wise with the noised video latent X_t
     before the first DiT block (VACE's own conditioning-concat pattern), rather
     than inventing a new injection mechanism the paper doesn't describe.
This is our design choice to fill a real gap in the paper, not something stated
in the text -- flagged here so it's not mistaken for a reported detail later.

## Open questions
- Public substitute for the 1.1M internal video corpus: using OpenVid-1M
  (NJU-PCALab, CC-BY-4.0, ~1M text-video clips, closest public scale match to the
  paper's 1.1M), run through DEVA for per-object masks.
- Exact random condition-dropping schedule (paper says it happens, not the rate).
- Box smoothing/noise augmentation parameters (paper says it happens, not the
  specifics).

## How training pairs are actually built from DEVA masks (our design, not stated
## verbatim in the paper, but the standard construction for this class of model)
Training needs M_obj (where the object should end up) and M_inpaint (where it
originally was) for a video where the *ground truth output* is just the original,
unedited clip. The natural construction: DEVA gives a real object's per-frame mask
-> convert to a real box trajectory B_real. Use B_real (with the paper's stated
smoothing+noise augmentation applied) as BOTH M_inpaint and M_obj during training.
The model is trained to erase the object at B_real and resynthesize it back at
B_real, matching the real ground-truth frames -- i.e. training never sees a case
where M_obj differs from M_inpaint. At inference, TRACE swaps in a genuinely
different M_obj (from Stage 1's predicted new trajectory) while M_inpaint stays
at the object's real original location, which is what actually produces motion
editing rather than reconstruction. This is the same reconstruction-as-training
/ editing-as-inference pattern most mask-conditioned video inpainting models use;
implemented in stage2_data_pipeline.py.
