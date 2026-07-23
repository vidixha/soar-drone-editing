# FINDINGS.md, Aerial Cross-View Box Propagation POC

All numbers below are from `results/protocol_a_summary.json`, produced by
`src/run_protocol_a.py` + `src/analyze_results.py` on the VisDrone-MOT validation split
(7 scenes, 2846 frames total, see NOTES.md for dataset verification). Protocol A only:
static objects, camera-motion compensation. Protocol B and method 4 (depth-based
warping) were not completed in this pass, see Limitations.

**This document was corrected after review.** The first draft's headline claim
("geometry wins outright against TRACE") was not supported by what was actually run and
has been rewritten below. What changed and why is at the end of this section, kept for
the record rather than silently edited away.

## What was actually tested, stated precisely

Four methods were planned. Two were built and are meaningfully different:
**static box** (the floor) and **homography propagation** (chained and direct
variants). The other two are not comparisons this run can make:

- **Linear interpolation** degenerates to the static box by mathematical construction in
  Protocol A (only one reference point is given, nothing to interpolate toward). It is
  not a second data point, it is the same number under a different label.
- **Depth-based warping** (TRACE's third baseline) was not implemented. No GPU was
  available for DROID-SLAM.
- **TRACE's learned cross-view module** was never run. No result here says anything about
  how a learned model compares to geometry, on any domain.

So the actual comparison in this experiment is **background-homography propagation
versus doing nothing, on 110 static tracks across 7 VisDrone-MOT scenes.** That is a
narrower and more defensible claim than "geometry beats TRACE," and it is the one this
document now makes.

## Headline finding, corrected

**On this footage, background-homography propagation substantially reduces box
placement error versus no compensation** (mean IoU 0.358 chained / 0.338 direct versus
0.146 for the static-box floor; mean center error drops from 136px to 36px, a 3.8x
reduction). The acceptance control passes: the floor is clearly beaten, so the harness is
measuring something real.

**That is the full claim.** It is not evidence that geometry "wins" against TRACE, a
learned model, or MegaSAM/depth-based warping, because none of those were run. It is not
evidence the result would hold under aggressive camera motion, because this dataset does
not contain any (see Limitations). And, more importantly than either of those, **it is
not evidence the output is good enough to use.**

## Why 0.358 mean IoU is not a success number for this project's actual goal

The wider project's purpose is generating usable ground-truth boxes for training aerial
perception models. Median IoU across all homography-chained predictions is **0.329**,
below the standard 0.5 IoU detection-match threshold. Framed as "geometry wins," 0.358
mean IoU reads like a positive result. Framed against the actual goal, it means that in
the typical case, the predicted box does not even clear the bar normally used to call a
detection correct.

**The honest reading is close to the opposite of the original headline: geometric
propagation alone, even in this dataset's easy low-motion regime, does not produce boxes
of usable label quality on aerial video.** It is a large, measurable improvement over
doing nothing, and it is not adequate on its own. Both statements are true and neither
should be dropped in favor of the other.

## A circularity in how static tracks were selected, not previously disclosed

Static tracks were selected using the proxy specified in TASK.md: a track qualifies if
its box motion is well explained by a background homography fit between consecutive
appearances (see `src/static_track_selector.py`). Homography propagation is then
evaluated on exactly that same set of tracks.

**The selection criterion and the method under test are the same model family.** Tracks
that a similarity-transform homography fits poorly (independent of whether they are
actually static in the world) are filtered out before evaluation ever happens. This
mechanically inflates homography's apparent accuracy and, by the same mechanism,
deflates the static-box control's relative disadvantage on exactly the cases the
selection process kept out. The mid-session bug fix documented below (switching from a
long-chain to a local-step residual) reduced the amount of *chain-length-induced* noise
in this proxy, but it did not, and could not, remove this circularity: it is still the
same model family doing the selecting and the scoring.

This proxy is what TASK.md specified, so this is a task-design limitation, not an
implementation error, and it needs to be named as one. **The correct fix is an
independent, non-homography-based way of confirming a track is static** (for example: an
appearance-based check, a manual spot-check subsample, or cross-referencing against scene
metadata), and re-running the comparison on tracks selected that way. Until that is done,
homography's advantage over the static-box floor in this report should be read as an
upper bound, not a clean estimate.

## Per-scene results, not strata

The first draft of this document bucketed 7 scenes into 3 "motion magnitude" and 3
"altitude change" terciles and presented the result as stratified findings. **With
roughly 2-3 scenes per bucket, those are not strata, they are scene identities wearing a
statistical label**, and the resulting claims did not hold up: the motion-magnitude
buckets gave IoU 0.233 / 0.580 / 0.439 (low/medium/high) and altitude gave
0.391 / 0.214 / 0.481, neither of which is monotonic, and the original "geometry helps
most where it's needed most" claim was written by comparing the low and high buckets
while stepping over a medium bucket that contradicts it. That claim is retracted.

What the data actually supports is a **per-scene table**, reported as observations
about 7 individual scenes, not as a trend across a motion-magnitude variable:

| scene | n tracks | translation (px/frame) | static IoU | homography_chained IoU | homography_direct IoU |
|---|---|---|---|---|---|
| uav0000339_00001_v | 5 | 0.81 | 0.079 | 0.577 | 0.277 |
| uav0000268_05773_v | 7 | 0.87 | 0.328 | 0.246 | 0.164 |
| uav0000086_00000_v | 16 | 1.33 | 0.072 | 0.157 | 0.199 |
| uav0000117_02622_v | 30 | 1.67 | 0.282 | 0.426 | 0.398 |
| uav0000305_00000_v | 17 | 2.06 | 0.184 | 0.681 | 0.765 |
| uav0000137_00458_v | 6 | 2.22 | 0.061 | 0.681 | 0.581 |
| uav0000182_00000_v | 29 | 2.38 | 0.063 | 0.378 | 0.370 |

Sorted by measured translation. There is no clean trend: the two lowest-motion scenes
give the best (0.577) and one of the worst (0.246) chained-homography results in the
whole set. Scene content (crowd density, background texture available for ORB matching,
how many static tracks happened to qualify) plausibly dominates over motion magnitude as
the driver of variance here, but that is a hypothesis, not something this data can
confirm with n=7. Occlusion and motion-type breakdowns from the original draft have the
same n=7-scenes-is-not-a-stratum problem and are not repeated here as findings; the raw
numbers remain in `results/protocol_a_summary.json` for anyone who wants them, labeled as
what they are.

## The drift curve: what holds up, corrected

The bin previously labeled "offset=0" does not contain any track evaluated at its literal
reference frame. By construction (`src/run_protocol_a.py`, the reference frame itself is
never scored), the minimum possible `frame_offset` is 5, the key-frame stride. The bin
labeled 0 (bin width 10) actually spans offsets 5-9. This is a mislabeled axis, not a
scoring bug: static box scoring 0.681 (not 1.0) and homography scoring 0.758 at a 5-9
frame real gap is a plausible, real number, not evidence of a transform-application error.
The bin label should be read as "5-9 frames after reference," not "identical frame."

What still holds from the original drift analysis:

- Static box degrades fast: from 0.681 IoU at 5-9 frames to 0.280 by ~30 frames offset,
  under 0.10 by ~90 frames.
- Chained homography holds a real advantage out to roughly 150-200 frames, then degrades
  further, converging to and briefly dipping below the static-box floor by roughly
  400-460 frames offset. Consistent with compounding drift in a chain of dozens to
  hundreds of composed local transforms.
- Direct (non-chained) homography degrades faster and more erratically past roughly 150
  frames, sometimes failing outright (IoU near 0.00) at long baselines where ORB matching
  between temporally distant frames has little shared content. The chained-vs-direct
  crossover around 150-200 frames is a genuine, somewhat surprising result, but the tail
  bins (offset 300+) rest on as few as 6-9 frame instances drawn from a handful of
  long-lived tracks concentrated in two sequences. **State this as indicative, not a
  precise failure threshold.**

## Scale error: the claim that homography "earns its keep" on scale is wrong, corrected

Overall mean scale deviation (|predicted area / ground-truth area - 1|): static 0.238,
homography_chained 0.219, homography_direct 0.280. **Direct homography is worse than
doing nothing on scale. Chained homography is only marginally better than doing nothing
(0.219 vs 0.238), not a clear win.** The original document's altitude-change section
claimed homography "earns its keep" by modeling scale explicitly; that claim is not
supported by the overall numbers and is retracted. A similarity transform does model
scale in principle, but the ORB-based scale estimate is evidently noisy enough in
practice that it does not reliably outperform assuming no scale change at all. This is
worth investigating further (is the scale term in `geometry.transform_scale` well
conditioned, or is it dominated by RANSAC inlier noise), not asserted as a solved
strength.

## What is solid

- **Center error**: 136px (static) to 36px (chained homography), a clean 3.8x reduction,
  and a more interpretable number than IoU for small aerial objects where a few pixels of
  box-size mismatch swings IoU a lot.
- **The chained-vs-direct crossover around 150-200 frames offset** is a real, specific,
  somewhat surprising pattern (direct avoids chain drift but pays for it with match
  failure at long temporal baselines), reported with the caveat that the far tail is
  thin.
- **The acceptance control passed**: static box is clearly and consistently beaten by
  both homography variants, confirming the metrics harness detects failure as designed.

## Honest limitations

- **The circularity in static-track selection**, detailed above. This is the most
  important open issue in this document, not a footnote.
- **The dataset cannot answer the question the task actually asked, and this is the
  central limitation, not a minor caveat.** Every sequence in VisDrone-MOT's validation
  split shows sub-3px/frame translation and sub-0.15deg/frame rotation (NOTES.md 1.3):
  this is hovering surveillance footage, not translating 6-DoF drone flight. The question
  was whether geometric box propagation survives real drone motion; this data is
  incapable of testing that regime, in either direction. Every number in this document is
  measured in the easy case. **Before any further reanalysis of these 7 scenes, the
  priority is footage with real camera translation and object annotations.** UAVDT is the
  concrete next candidate: it has MOT annotations plus explicit camera-view and altitude
  attributes, and is the more plausible source of genuine translating-camera sequences.
  This is a bigger lever than anything else in this document.
- **Dataset size: 7 sequences, not the 10 the acceptance criteria asked for.** This is
  the entire VisDrone-MOT validation split; there is no way to reach 10 from this dataset
  alone. 110 static tracks qualified, exceeding the 30-track minimum, but see the
  circularity note on what "qualified" means here.
- **No rotation-dominant sequence exists in this dataset**, so the motion-type
  stratification TASK.md asked for (translation vs rotation, likely where homography and
  depth warping diverge) could not be tested at all, and is not reported as a per-scene
  table either since it never had two categories to compare.
- **Key-frame subsampling, not per-frame evaluation.** Stride 5 (`KEY_FRAME_STRIDE` in
  `src/scene_transforms.py`), for resource reasons (CPU only, no GPU, this session).
  The drift curve has 5-frame temporal resolution, not 1-frame.
- **Method 4 (depth-based warping) was not implemented.** DROID-SLAM requires CUDA with
  no CPU fallback found in the official repo or any fork; this machine had no GPU. An
  attempt to use Kaggle's free GPU tier for this was started and blocked on an expired
  API credential; getting a valid credential and re-attempting DROID-SLAM (or the
  previously scoped essential-matrix substitute) is the next concrete step for this
  method specifically, independent of the dataset issue above.
- **Protocol B (moving-object path propagation) was not built.**
- **Occlusion and motion-type breakdowns from the first draft are not repeated as
  findings**, for the same n=7-scenes reason as the motion-magnitude/altitude strata.

## What changed in this revision, and why

The first draft claimed a direct, favorable comparison against TRACE that the experiment
did not run (no learned module, no MegaSAM/depth baselines). It presented 0.358 mean IoU
as a win without weighing it against the 0.5 IoU bar that would make a box usable as a
training label, and without noting the median (0.329) is below that bar. It stratified 7
scenes into 3-way buckets and drew a monotonic trend from non-monotonic numbers. It
missed that static-track selection and the evaluated method are the same model family. It
mislabeled a drift-curve bin in a way that looked like it might be a scoring bug. And it
claimed a scale-modeling advantage for homography that the overall numbers contradict.
Every one of those is corrected above. The center-error result, the acceptance-control
pass, and the chained-vs-direct crossover survive review and are kept as the solid
findings of this pass.
