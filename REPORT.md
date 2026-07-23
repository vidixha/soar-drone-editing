# Running Report, Plain Language

Updated as we go. This is the "what's happening and why" doc. For technical detail, see
NOTES.md and FINDINGS.md in this folder.

## The goal, in one sentence

Insert a fake moving object (a person, a cyclist) into real drone video so it looks real,
and get free, correct training labels out of it for free. **The final deliverable is a
plausible-looking video.**

## Why we're not building that directly yet

Before you can draw a fake object into a video, you need to know exactly where it should
go in every single frame, correcting for the fact the drone camera itself is moving. Get
that wrong and the inserted object visibly drifts or floats. So we're first proving out
the "where does it go" piece using real objects already in the footage (so we have ground
truth to check against), before drawing anything fake in.

## Status as of now

**Done:**
- Built a working pipeline: given one box on a real object, predict where that box should
  be in later frames using only camera-motion math (no AI). Tested on ~2800 frames of
  real drone footage (VisDrone dataset).
- Result: the math-only approach clearly beats "do nothing," often by a lot, but not
  reliably, and it can get worse the longer the video runs.
- Made two short videos showing this visually (boxes drawn on real footage) and one
  "practice insertion" video (a placeholder cartoon shape pasted into the footage using
  the same math), so the effect is visible, not just numbers.
- Independent review of the first write-up caught real problems (an unfair test setup, a
  misleading stat, an overclaimed headline). All fixed, documented plainly, nothing
  swept under the rug.

**Since then, in order:**

1. **Second dataset (UAVDT) confirms the geometry result.** 46 real clips (vs 7 before),
   424,264 measurements. Same pattern as the first dataset: on clips with real drone
   motion, the math-only approach beats doing nothing by 2-7x; across the full mix it's
   messier, with no clean "more motion = more benefit" line. Two datasets now say the
   same thing, which makes it a sturdier finding.
2. **DROID-SLAM: moved to Modal, works.** Kaggle's free GPU pool handed us an old card
   (a "P100") that Kaggle's own pre-installed software no longer supports at all, an
   unfixable-from-our-side dead end after 10 tries. On Modal we can request a specific
   GPU (a T4) every time. First real run: 10 sample frames, real camera pose recovered.
   Scaled up to 100 real frames from the same clip used elsewhere in this project: it
   kept 4 real keyframes and recovered a genuine, non-straight-line camera path, saved as
   a plot and a short video of the real keyframes in sequence.
3. **VACE: the actual breakthrough result.** Three attempts, each answering a different
   question:
   - Attempt 1 (arbitrary box, generic prompt): meaningless blur. Proved the tool runs.
   - Attempt 2 (better prompt, still an arbitrary fixed box): more coherent-looking, but
     it quietly erased a real cyclist instead of adding anything. Wrong behavior, but a
     more informative failure.
   - **Attempt 3 (a REAL per-frame path, from actual ground-truth tracking of a moving
     car in this footage, not a made-up box): produced a genuine, recognizable car,
     correctly positioned, consistent across all 9 frames.** This is the first time in
     this whole project that "insert something and have it look right" actually worked.
     Confirmed by comparing the exact masked region against the source frame directly,
     not just eyeballing the whole picture.

This is the answer to the open question from before: **accurate, real geometric
placement is what was missing.** Arbitrary boxes gave arbitrary (bad) results; a real
computed path gave a real, correct-looking result. That is direct evidence the two halves
of this project (the geometry, and the AI insertion) can work together, not just
separately.

Getting attempt 3 to run took real troubleshooting on Modal too: a build-isolation
issue, a stray dependency (`flash_attn`) that isn't needed and doesn't build cleanly, a
missing `matplotlib`, a HuggingFace rate limit (fixed by using an authenticated token),
running out of GPU memory (fixed by matching the clip length to the model's settings
instead of using an unrelated default), and a timeout that was just set too short for how
long the actual generation takes (about 18 minutes). None of these were dead ends, each
was a specific, fixable thing.

## What's next

1. Update `FINDINGS.md` with the UAVDT confirmation (not yet folded in there).
2. Run the same "real path" VACE test on a few more real tracked objects, to confirm
   attempt 3 wasn't a lucky one-off.
3. Compare DROID-SLAM's recovered camera path against our own simple-math path on the
   exact same clip, the one direct comparison that's still never been run.
4. Consider whether VACE's largest model (not the 1.3B preview used so far) does even
   better, now that we know the *approach* works.
