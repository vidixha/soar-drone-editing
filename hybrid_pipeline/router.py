"""SOAR NL router: natural-language instruction -> composed edited video.

Pipeline (ChatSim-style orchestration, our training-free modules):
  parse   : NL -> ordered op-list (rule-based; compound instructions supported)
  schedule: canonical order  remove -> insert -> trajectory
            (content/geometry edits before appearance edits, when those exist)
  dispatch: each op -> its module in modules.py
  render  : compose -> one output video

Usage:
  python router.py --clip clip.mp4 --depth depth.png \
      --instruction "remove the cars and orbit left" --out out.mp4 --trajectory-gpu

The parser is intentionally a thin NL->JSON step. A rule-based parser handles
single and compound instructions; an LLM backend (Anthropic API) is a drop-in
replacement for free-form language (parse_llm stub below) and is the only place
an LLM would enter -- it never touches pixels.

This branch (w4/hybrid_pipeline) ships the NL router, object removal, and
trajectory editing. Weather modules and the insertion prototype live on a
separate branch.
"""
import argparse, re, json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modules as M

MODULE_ORDER = {"remove": 0, "insert": 1, "trajectory": 2}

def parse(text):
    """NL -> ordered op-list. Rule-based; deterministic; handles compound."""
    t = text.lower(); ops = []
    if re.search(r"\b(remove|delete|erase|get rid of|clear|take out)\b", t):
        m = re.search(r"(?:remove|delete|erase|clear|take out)\s+(?:the\s+|all\s+)?([a-z]+)", t)
        ops.append({"module": "remove", "params": {"target": m.group(1) if m else "objects"}})
    m = re.search(r"\b(?:insert|add|place|put)\s+(?:a\s+|another\s+)?(car|vehicle|truck|person|object)\b", t)
    if m: ops.append({"module": "insert", "params": {"object": m.group(1)}})
    for kw, mv in [("orbit","orbit"),("circle","orbit"),("pan","pan"),("dolly","dolly"),
                   ("fly","fly"),("zoom","zoom"),("crane","crane")]:
        if re.search(rf"\b{kw}\b", t):
            d = next((dd for dd in ["left","right","forward","backward","in","out","up","down"]
                      if re.search(rf"\b{dd}\b", t)), "left")
            ops.append({"module": "trajectory", "params": {"motion": mv, "dir": d}}); break
    return ops

def parse_llm(text):
    """Drop-in LLM parser for free-form language (Anthropic API). Not required for
    the demo; the rule-based parse() covers single + compound instructions."""
    raise NotImplementedError("wire an LLM here for open-vocabulary instructions")

def schedule(ops):
    """Canonical order: content/geometry edits before appearance edits."""
    return sorted(ops, key=lambda o: MODULE_ORDER[o["module"]])

def pose_of(motion, direction):
    """Map an NL motion+direction to a TrajectoryCrafter target pose
    [theta(tilt), phi(pan), r(dolly), x, y]."""
    sgn = -1 if direction in ("left", "in", "forward", "up") else 1
    if motion in ("orbit", "circle"): return (0.0, sgn*25.0, 0.15, 0.0, 0.0)
    if motion == "pan":               return (0.0, sgn*22.0, 0.0, 0.0, 0.0)
    if motion in ("dolly", "fly", "zoom"):
        if direction in ("up", "down"): return (sgn*15.0, 0.0, 0.0, 0.0, 0.0)
        return (0.0, 0.0, 0.3 if direction in ("in","forward") else -0.3, 0.0, 0.0)
    return (0.0, 20.0, 0.2, 0.0, 0.0)

def _depth_seq_gpu(frames, fps):
    """Trace Anything: one feed-forward pass over the WHOLE clip, returning
    genuinely per-frame, temporally-consistent depth (T,H,W) instead of a
    single snapshot. Replaces both the old initial single-frame depth load
    AND the old post-trajectory refresh -- same call, either place. Validated:
    mean frame-to-frame depth change 0.004 on a static hover clip, visually
    tracks real scene structure including small moving-object signatures.
    Model weights are CC-BY-NC-4.0."""
    import tempfile, io, modal, cv2
    tmp = tempfile.mktemp(suffix=".mp4"); M.save_video(frames, tmp, fps)
    fn = modal.Function.from_name("trace-anything-depth", "depth_sequence_of_bytes")
    npz_bytes = fn.remote(open(tmp, "rb").read())
    seq = np.load(io.BytesIO(npz_bytes))["depth"].astype(np.float32)
    # Trace Anything's own preprocessing resizes to its native scale
    # (long_side=512) -- resize back to our actual working resolution, same
    # as load_depth() already does for the single-frame path. Missing this
    # step caused an IndexError: particle x-positions ranged over the real
    # frame width (1280) while the un-resized depth was only 512 wide.
    H, W = frames[0].shape[:2]
    if seq.shape[1:] != (H, W):
        seq = np.stack([cv2.resize(f, (W, H)) for f in seq], axis=0)
    return seq

def _traj_gpu(frames, motion, direction, fps, sample_h=None, sample_w=None, use_trace_anything=False):
    """Fire a real TrajectoryCrafter render on Modal for the current frames,
    then refresh depth for the NEW viewpoint -- the old depth no longer
    matches scene geometry after the camera moves. use_trace_anything selects
    the per-frame backbone (_depth_seq_gpu) over the older single-frame
    Depth Anything V2 refresh.

    sample_h/sample_w default to None (unset): explicitly passing
    --sample_size (tried both 720x1280 and 480x848) reliably crashes
    TrajectoryCrafter's own --mode gradual compositing with a hardcoded
    ~384px internal reference -- confirmed twice, not a resolution-tuning
    problem. Leaving it unset uses inference.py's own working default:
    softer output (documented, known limitation) but doesn't crash."""
    import tempfile, modal
    tmp = tempfile.mktemp(suffix=".mp4"); M.save_video(frames, tmp, fps)
    video_bytes = open(tmp, "rb").read()
    pose = pose_of(motion, direction)
    fn = modal.Function.from_name("trajcrafter-test", "render_traj")
    kwargs = {"video_length": 49}
    if sample_h is not None and sample_w is not None:
        kwargs["sample_h"] = sample_h; kwargs["sample_w"] = sample_w
    out_bytes = fn.remote(video_bytes, *pose, **kwargs)
    res = tempfile.mktemp(suffix=".mp4"); open(res, "wb").write(out_bytes)
    new_frames = M.load_clip(res)

    if use_trace_anything:
        new_depth = _depth_seq_gpu(new_frames, fps)
    else:
        depth_fn = modal.Function.from_name("depth-extract", "depth_of_bytes")
        new_video = tempfile.mktemp(suffix=".mp4"); M.save_video(new_frames, new_video, fps)
        depth_png = depth_fn.remote(open(new_video, "rb").read())
        depth_path = tempfile.mktemp(suffix=".png"); open(depth_path, "wb").write(depth_png)
        new_depth = M.load_depth(depth_path, new_frames[0].shape[:2])
    return new_frames, pose, new_depth

TRAJ_CACHE_VIDEO = "/tmp/soar_router_traj_cache.mp4"
TRAJ_CACHE_DEPTH = "/tmp/soar_router_traj_cache_depth.npz"

def execute(clip, ops, depth_path, out_path, fps=30, trajectory_gpu=False, use_trace_anything=False,
            reuse_trajectory_cache=False, save_trajectory_cache=True):
    """use_trace_anything=False (default): single Depth Anything V2 snapshot
    loaded once, refreshed via one more single-frame call after trajectory.
    use_trace_anything=True: initial depth AND the post-trajectory refresh
    both come from Trace Anything's per-frame sequence instead -- genuinely
    correct depth for every frame (must be explicitly opted into; never fires
    by default, since it costs an extra A100-80GB GPU call each time).

    reuse_trajectory_cache: skip the expensive TrajectoryCrafter GPU call
    entirely and load the last cached post-trajectory frames+depth instead --
    useful for iterating on anything downstream of trajectory without
    re-paying for a ~5-6min A100 render every time. save_trajectory_cache:
    persist a fresh GPU result for later reuse (on by default whenever a real
    render happens)."""
    frames = M.load_clip(clip)
    depth = _depth_seq_gpu(frames, fps) if use_trace_anything else M.load_depth(depth_path, frames[0].shape[:2])
    trace = []
    for op in ops:
        mod, p = op["module"], op["params"]
        if mod == "remove":
            frames, _ = M.remove_objects(frames); trace.append(f"remove({p['target']}) ✓")
        elif mod == "trajectory":
            if reuse_trajectory_cache and os.path.exists(TRAJ_CACHE_VIDEO) and os.path.exists(TRAJ_CACHE_DEPTH):
                frames = M.load_clip(TRAJ_CACHE_VIDEO)
                depth = np.load(TRAJ_CACHE_DEPTH)["depth"].astype(np.float32)
                trace.append(f"trajectory({p['motion']},{p['dir']}) -> reused cached result (no GPU spend)")
            elif trajectory_gpu:
                frames, pose, depth = _traj_gpu(frames, p["motion"], p["dir"], fps, use_trace_anything=use_trace_anything)
                backbone = "Trace Anything (per-frame)" if use_trace_anything else "Depth Anything V2 (single-frame)"
                trace.append(f"trajectory({p['motion']},{p['dir']}) -> TrajectoryCrafter pose={pose} ✓ (GPU, depth refreshed via {backbone})")
                if save_trajectory_cache:
                    M.save_video(frames, TRAJ_CACHE_VIDEO, fps)
                    np.savez_compressed(TRAJ_CACHE_DEPTH, depth=depth.astype(np.float32))
                    trace.append(f"    (cached post-trajectory result for reuse: {TRAJ_CACHE_VIDEO})")
            else:
                trace.append(f"trajectory({p['motion']},{p['dir']}) -> GPU render [use --trajectory-gpu to fire]")
        elif mod == "insert":
            trace.append(f"insert({p['object']}) -> not implemented on this branch [prototype lives elsewhere]")
    M.save_video(frames, out_path, fps)
    return trace

def run(instruction, clip, depth, out, trajectory_gpu=False, use_trace_anything=False,
        reuse_trajectory_cache=False):
    ops = parse(instruction); plan = schedule(ops)
    print(f'\n  INSTRUCTION: "{instruction}"')
    print(f"  PARSED  : {json.dumps(ops)}")
    print(f"  SCHEDULE: {json.dumps(plan)}   (remove→insert→trajectory)")
    print( "  EXECUTE :")
    for line in execute(clip, plan, depth, out, trajectory_gpu=trajectory_gpu, use_trace_anything=use_trace_anything,
                        reuse_trajectory_cache=reuse_trajectory_cache):
        print(f"      - {line}")
    print(f"  OUTPUT  : {out}\n")
    return plan

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--depth", default="")
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trajectory-gpu", action="store_true", help="fire real TrajectoryCrafter render on Modal")
    ap.add_argument("--use-trace-anything", action="store_true",
                     help="use Trace Anything's per-frame depth sequence instead of single-snapshot Depth Anything V2 (extra A100-80GB GPU call(s), opt-in only)")
    ap.add_argument("--reuse-trajectory-cache", action="store_true",
                     help="skip the GPU trajectory render, reuse the last cached post-trajectory result")
    a = ap.parse_args()
    run(a.instruction, a.clip, a.depth, a.out, trajectory_gpu=a.trajectory_gpu, use_trace_anything=a.use_trace_anything,
        reuse_trajectory_cache=a.reuse_trajectory_cache)
