"""Edit modules for the SOAR aerial video editor. Each is TRAINING-FREE and
operates on a list of BGR frames (+ optional depth). Consolidated from the
validated per-module prototypes:

  remove_objects  : classical background-reveal (single sharp plate, drift-aligned)

The router (router.py) calls these; it never re-implements image ops.

(Weather modules -- fog/rain/snow/sandstorm -- and the insertion prototype
live on a separate branch and are not included here.)
"""
import cv2, numpy as np, os, subprocess, tempfile

# ---------------------------------------------------------------- io helpers
def load_clip(path, max_side=1280):
    cap = cv2.VideoCapture(path); frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        h, w = f.shape[:2]; s = max_side / max(h, w)
        frames.append(cv2.resize(f, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA))
    cap.release()
    return frames

def save_video(frames, path, fps=30):
    d = tempfile.mkdtemp()
    for i, fr in enumerate(frames): cv2.imwrite(f"{d}/{i:05d}.png", fr)
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", f"{d}/%05d.png", "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-crf", "20", path], check=True, capture_output=True)
    subprocess.run(["rm", "-rf", d])

def load_depth(path, shape):
    """Load a precomputed Depth-Anything disparity map (higher=closer), resized."""
    H, W = shape
    if path and os.path.exists(path):
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        d = cv2.resize(d, (W, H)); return cv2.GaussianBlur(d / 65535.0, (0, 0), 2)
    # proxy: vertical gradient matching REAL depth convention (near=high at
    # bottom, far=low at top) -- must match loaded depth's sign convention or
    # downstream consumers would silently invert.
    return np.repeat(np.clip(np.linspace(0, 1, H)*1.15, 0, 1)[:, None], W, 1)

def load_depth_sequence(path, shape):
    """Load a Trace Anything per-frame depth stack (.npz, key 'depth', shape
    (T,H,W), globally-normalized [0,1], higher=closer) -- see
    modal/trace_anything_depth.py. Falls back to a broadcastable single-frame
    depth (proxy or file) if no sequence is available."""
    if path and os.path.exists(path) and path.endswith(".npz"):
        seq = np.load(path)["depth"].astype(np.float32)
        H, W = shape
        if seq.shape[1:] != (H, W):
            seq = np.stack([cv2.resize(f, (W, H)) for f in seq], axis=0)
        return seq
    return load_depth(path, shape)  # single-frame fallback

def _depth_at(depth, t):
    """depth is either a single (H,W) array (same map reused for every frame)
    or a (T,H,W) sequence (Trace Anything, per-frame correct)."""
    if depth.ndim == 3:
        return depth[min(t, depth.shape[0]-1)]
    return depth

# ---------------------------------------------------------------- removal
def _fg_masks(frames, med, thr=22, dil=13, min_area=100):
    out = []
    for f in frames:
        d = cv2.absdiff(f, med).max(2); _, m = cv2.threshold(d, thr, 255, cv2.THRESH_BINARY)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8), 3)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8); keep = np.zeros_like(m)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] >= min_area: keep[lab == i] = 255
        out.append(cv2.dilate(keep, np.ones((dil, dil), np.uint8), 1))
    return out

def remove_objects(frames):
    """Remove moving objects via a single sharp static plate + drift-aligned fill.
    Assumes a near-static (hover) camera -- the tractable case."""
    H, W = frames[0].shape[:2]; T = len(frames)
    med = np.median(np.stack(frames).astype(np.float32), 0).astype(np.uint8)
    FG = _fg_masks(frames, med)
    SB = (int(H*0.12), int(H*0.29), int(W*0.44), int(W*0.59))
    gref = [cv2.cvtColor(f[SB[0]:SB[1], SB[2]:SB[3]], cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    def drift(a, b):
        (dx, dy), _ = cv2.phaseCorrelate(gref[a], gref[b]); return dx, dy
    def shift(img, dx, dy):
        return cv2.warpAffine(img, np.float32([[1,0,dx],[0,1,dy]]), (W,H),
                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    base = int(np.argmin([(m > 0).sum() for m in FG]))
    plate = frames[base].copy(); bm = FG[base]
    if bm.sum():
        n, lab, st, _ = cv2.connectedComponentsWithStats(bm, 8)
        for i in range(1, n):
            comp = (lab == i).astype(np.uint8)*255; ys, xs = np.where(comp > 0)
            y1, y2, x1, x2 = ys.min(), ys.max()+1, xs.min(), xs.max()+1; bestk, bs = 0, 1e18
            for k in range(T):
                if k == base: continue
                sc = (FG[k][y1:y2, x1:x2] > 0).sum() - 0.001*abs(k-base)
                if sc < bs: bs, bestk = sc, k
            dx, dy = drift(base, bestk); don = shift(frames[bestk], dx, dy)
            cm = cv2.GaussianBlur(comp, (0,0), 1.5).astype(np.float32)[..., None]/255.0
            plate = (plate*(1-cm)+don*cm).astype(np.uint8)
    out = []
    for t in range(T):
        dx, dy = drift(t, base); pl = shift(plate, dx, dy)
        mf = cv2.GaussianBlur(FG[t], (0,0), 1.5).astype(np.float32)[..., None]/255.0
        out.append((frames[t]*(1-mf)+pl*mf).astype(np.uint8))
    return out, plate
