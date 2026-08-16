"""SOAR NL router: natural-language instruction -> composed edited video.

Pipeline (ChatSim-style orchestration, our training-free modules):
  parse   : NL -> ordered op-list (rule-based; compound instructions supported)
  schedule: canonical order  remove -> insert -> trajectory
            (content/geometry edits before appearance edits, when those exist)
  dispatch: each op -> its module in modules.py, or a GPUBackend for
            anything that needs a real render (see gpu_backend.py)
  render  : compose -> one output video

Usage:
  python router.py --clip clip.mp4 --depth depth.png \
      --instruction "remove the cars and orbit left" --out out.mp4 --trajectory-gpu

The parser is intentionally a thin NL->JSON step. A rule-based parser handles
single and compound instructions; an LLM backend (Anthropic API) is a drop-in
replacement for free-form language (parse_llm stub below) and is the only place
an LLM would enter -- it never touches pixels.

GPU work goes through gpu_backend.GPUBackend, not any specific provider --
this file never imports Modal (or anything else GPU-provider-specific)
directly. modal_backend.py ships the one concrete implementation used by
default; swap in another by implementing GPUBackend and passing
--backend <name> (after registering it in gpu_backend.get_backend()).

This branch (w4/hybrid_pipeline) ships the NL router, object removal, and
trajectory editing. Weather modules and the insertion prototype live on a
separate branch.
"""
import argparse, re, json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modules as M
from gpu_backend import get_backend

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
    """Small-LLM parser (Qwen2.5-0.5B-Instruct, CPU, ~1.6GB RAM) -- see
    llm_parser.py. Generalizes past parse()'s fixed keyword list (handles
    paraphrases, e.g. "get the vehicles out of the frame" with zero keyword
    overlap with parse()'s regex vocabulary) at the cost of ~15-25s of local
    CPU inference per call. Imported lazily so choosing the default regex
    parser never pulls in torch/transformers."""
    from llm_parser import parse_llm as _parse_llm
    return _parse_llm(text)

def schedule(ops):
    """Canonical order: content/geometry edits before appearance edits."""
    return sorted(ops, key=lambda o: MODULE_ORDER[o["module"]])

def pose_of(motion, direction):
    """Map an NL motion+direction to a target camera pose
    [theta(tilt), phi(pan), r(dolly), x, y]."""
    sgn = -1 if direction in ("left", "in", "forward", "up") else 1
    if motion in ("orbit", "circle"): return (0.0, sgn*25.0, 0.15, 0.0, 0.0)
    if motion == "pan":               return (0.0, sgn*22.0, 0.0, 0.0, 0.0)
    if motion in ("dolly", "fly", "zoom"):
        if direction in ("up", "down"): return (sgn*15.0, 0.0, 0.0, 0.0, 0.0)
        return (0.0, 0.0, 0.3 if direction in ("in","forward") else -0.3, 0.0, 0.0)
    return (0.0, 20.0, 0.2, 0.0, 0.0)

def _depth_seq_gpu(backend, frames, fps):
    """Per-frame, temporally-consistent depth for the WHOLE clip in one pass
    (e.g. Trace Anything via ModalBackend), instead of a single snapshot
    reused for every frame. Replaces both the initial depth load AND the
    post-trajectory refresh -- same call, either place. Validated: mean
    frame-to-frame depth change 0.004 on a static hover clip, visually
    tracks real scene structure including small moving-object signatures."""
    import tempfile, io, cv2
    tmp = tempfile.mktemp(suffix=".mp4"); M.save_video(frames, tmp, fps)
    npz_bytes = backend.depth_sequence(open(tmp, "rb").read())
    seq = np.load(io.BytesIO(npz_bytes))["depth"].astype(np.float32)
    # Some backbones (e.g. Trace Anything) resize internally to their own
    # native processing scale -- resize back to our actual working
    # resolution, same as load_depth() does for the single-frame path.
    # Skipping this caused an IndexError: particle x-positions ranged over
    # the real frame width while the un-resized depth was narrower.
    H, W = frames[0].shape[:2]
    if seq.shape[1:] != (H, W):
        seq = np.stack([cv2.resize(f, (W, H)) for f in seq], axis=0)
    return seq

def _traj_gpu(backend, frames, motion, direction, fps, sample_h=None, sample_w=None, use_trace_anything=False):
    """Fire a real trajectory render via the backend for the current frames,
    then refresh depth for the NEW viewpoint -- the old depth no longer
    matches scene geometry after the camera moves. use_trace_anything selects
    the per-frame depth backbone (_depth_seq_gpu) over the backend's
    single-frame depth snapshot for the refresh.

    sample_h/sample_w default to None (unset): explicitly requesting an
    output resolution reliably crashes TrajectoryCrafter's own --mode
    gradual compositing (the reference backend) with a hardcoded ~384px
    internal reference -- confirmed at two different resolutions, not a
    tuning problem. Leaving it unset uses the backend's own working
    default: softer output (documented, known limitation) but doesn't crash."""
    import tempfile
    tmp = tempfile.mktemp(suffix=".mp4"); M.save_video(frames, tmp, fps)
    video_bytes = open(tmp, "rb").read()
    pose = pose_of(motion, direction)
    out_bytes = backend.render_trajectory(video_bytes, *pose, video_length=49,
                                          sample_h=sample_h, sample_w=sample_w)
    res = tempfile.mktemp(suffix=".mp4"); open(res, "wb").write(out_bytes)
    new_frames = M.load_clip(res)

    if use_trace_anything:
        new_depth = _depth_seq_gpu(backend, new_frames, fps)
    else:
        new_video = tempfile.mktemp(suffix=".mp4"); M.save_video(new_frames, new_video, fps)
        depth_png = backend.depth_single_frame(open(new_video, "rb").read())
        depth_path = tempfile.mktemp(suffix=".png"); open(depth_path, "wb").write(depth_png)
        new_depth = M.load_depth(depth_path, new_frames[0].shape[:2])
    return new_frames, pose, new_depth

TRAJ_CACHE_VIDEO = "/tmp/soar_router_traj_cache.mp4"
TRAJ_CACHE_DEPTH = "/tmp/soar_router_traj_cache_depth.npz"

def execute(clip, ops, depth_path, out_path, fps=30, trajectory_gpu=False, use_trace_anything=False,
            reuse_trajectory_cache=False, save_trajectory_cache=True, backend=None):
    """backend: a gpu_backend.GPUBackend instance (defaults to the Modal
    backend if not given -- see gpu_backend.get_backend()). Only constructed
    lazily, and only if a GPU op actually needs to fire, so a removal-only
    instruction never touches it and never needs a GPU provider configured.

    use_trace_anything=False (default): single depth snapshot loaded once,
    refreshed via one more single-frame call after trajectory.
    use_trace_anything=True: initial depth AND the post-trajectory refresh
    both come from the per-frame depth sequence instead -- genuinely correct
    depth for every frame (must be explicitly opted into; never fires by
    default, since it costs an extra GPU call).

    reuse_trajectory_cache: skip the expensive trajectory GPU call entirely
    and load the last cached post-trajectory frames+depth instead -- useful
    for iterating on anything downstream of trajectory without re-paying for
    a real render every time. save_trajectory_cache: persist a fresh GPU
    result for later reuse (on by default whenever a real render happens)."""
    frames = M.load_clip(clip)
    needs_gpu = use_trace_anything or (trajectory_gpu and any(o["module"] == "trajectory" for o in ops)
                                        and not (reuse_trajectory_cache and os.path.exists(TRAJ_CACHE_VIDEO)))
    if backend is None and needs_gpu:
        backend = get_backend()
    depth = _depth_seq_gpu(backend, frames, fps) if use_trace_anything else M.load_depth(depth_path, frames[0].shape[:2])
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
                if backend is None: backend = get_backend()
                frames, pose, depth = _traj_gpu(backend, frames, p["motion"], p["dir"], fps, use_trace_anything=use_trace_anything)
                backbone = "per-frame depth backend" if use_trace_anything else "single-frame depth backend"
                trace.append(f"trajectory({p['motion']},{p['dir']}) -> render ✓ (GPU, depth refreshed via {backbone})")
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
        reuse_trajectory_cache=False, backend_name="modal", parser="regex"):
    ops = (parse_llm if parser == "llm" else parse)(instruction)
    plan = schedule(ops)
    backend = get_backend(backend_name) if (trajectory_gpu or use_trace_anything) else None
    print(f'\n  INSTRUCTION: "{instruction}"')
    print(f"  PARSER  : {parser}")
    print(f"  PARSED  : {json.dumps(ops)}")
    print(f"  SCHEDULE: {json.dumps(plan)}   (remove→insert→trajectory)")
    print( "  EXECUTE :")
    for line in execute(clip, plan, depth, out, trajectory_gpu=trajectory_gpu, use_trace_anything=use_trace_anything,
                        reuse_trajectory_cache=reuse_trajectory_cache, backend=backend):
        print(f"      - {line}")
    print(f"  OUTPUT  : {out}\n")
    return plan

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--depth", default="")
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trajectory-gpu", action="store_true", help="fire a real trajectory render via the GPU backend")
    ap.add_argument("--use-trace-anything", action="store_true",
                     help="use the per-frame depth backend instead of the single-snapshot default (extra GPU call(s), opt-in only)")
    ap.add_argument("--reuse-trajectory-cache", action="store_true",
                     help="skip the GPU trajectory render, reuse the last cached post-trajectory result")
    ap.add_argument("--backend", default="modal", help="GPU backend to use (see gpu_backend.get_backend)")
    ap.add_argument("--parser", default="regex", choices=["regex", "llm"],
                     help="regex (default, free, instant, fixed keyword list) or llm "
                          "(Qwen2.5-0.5B-Instruct, CPU, ~15-25s/call, generalizes past exact keywords)")
    a = ap.parse_args()
    run(a.instruction, a.clip, a.depth, a.out, trajectory_gpu=a.trajectory_gpu, use_trace_anything=a.use_trace_anything,
        reuse_trajectory_cache=a.reuse_trajectory_cache, backend_name=a.backend, parser=a.parser)
