"""Edit modules for the SOAR aerial video editor. Each is TRAINING-FREE and
operates on a list of BGR frames (+ optional depth). Consolidated from the
validated per-module prototypes:

  remove_objects  : classical background-reveal (single sharp plate, drift-aligned)

The router (router.py) calls these; it never re-implements image ops.

Weather dispatch lives in weather.py (metric adapter for snow/rain/fog/
sandstorm). The insertion prototype is still a no-op on this branch.
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

def _ffmpeg():
    for candidate in (os.environ.get("FFMPEG"), "ffmpeg",
                      "/workspace/.tooling/ffmpeg/ffmpeg"):
        if not candidate:
            continue
        if candidate == "ffmpeg" or os.path.isfile(candidate):
            return candidate
    return "ffmpeg"


def save_video(frames, path, fps=30):
    d = tempfile.mkdtemp()
    for i, fr in enumerate(frames): cv2.imwrite(f"{d}/{i:05d}.png", fr)
    subprocess.run([_ffmpeg(), "-y", "-framerate", str(fps), "-i", f"{d}/%05d.png", "-vf",
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
def _mask_from_diff(frame, bg, thr=22, dil=13, min_area=100, max_area_frac=0.02, min_fill=0.35,
                    aspect_range=(0.3, 3.0), verify=None):
    """Foreground = compact, solid, car-shaped blobs only, not any diff from
    bg. A homography-aligned plate is planar-ground-only, so content that
    doesn't fit that model -- tall off-ground stuff (tree canopies), and
    plain sub-pixel alignment error over a large textured area (plowed
    fields, hedges) -- shows up as diff too. Fill ratio (area / bbox area)
    rejects thin, branch-like blobs; confirmed by direct inspection that
    fill ratio alone still lets through widespread scattered noise from
    field-texture misalignment (measured: ~95k false-positive px elsewhere
    in frame against ~300 true px on the actual target in one case), so
    also cap absolute size and require a plausible aspect ratio -- car-shaped
    means compact AND not a sliver, not just "solid for its own bounding box."

    `verify(crop_bgr) -> bool`, if given, is a further gate after the shape
    filter: the motion detector only ever proposes "compact solid moving
    thing," it has no notion of *which* one -- with multiple candidates
    (two cars, a car and a person) or an attribute in the instruction
    ("the black car"), shape alone can't tell them apart. See vlm_match.py."""
    H, W = frame.shape[:2]
    d = cv2.absdiff(frame, bg).max(2); _, m = cv2.threshold(d, thr, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    # Close just enough to fill small holes inside one object (e.g. a
    # windshield reflection gap in the car's own blob) -- not so much that
    # it bridges the gap to an unrelated, separate diff a few pixels away
    # (a tree-shadow blob sitting right next to the car), which would merge
    # them into one shape and fail the fill-ratio filter below entirely.
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), 1)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8); keep = np.zeros_like(m)
    max_area = max_area_frac * H * W
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area: continue
        bw, bh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        bbox_area = bw * bh
        if not (bbox_area and area / bbox_area >= min_fill): continue
        if not (aspect_range[0] <= bw / bh <= aspect_range[1]): continue
        if verify is not None:
            x, y = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
            bw, bh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
            pad = max(4, int(0.15 * max(bw, bh)))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(W, x + bw + pad), min(H, y + bh + pad)
            if not verify(frame[y0:y1, x0:x1]): continue
        keep[lab == i] = 255
    return cv2.dilate(keep, np.ones((dil, dil), np.uint8), 1)

def _estimate_step_homographies(grays):
    """Homography mapping frame i's pixels into frame i+1's coordinate
    system, for every adjacent pair -- estimated from background feature
    matches. RANSAC drops the moving object's own features as outliers as
    long as it covers a minority of the matched points -- true for a small
    object against a large textured field/road background. Falls back to
    identity (equivalent to the old static-camera assumption) when too few
    matches survive, e.g. a near-uniform patch of sky.

    Only ever matched between ADJACENT frames, never far ones: a tracking
    shot moves a lot over a whole clip, and matching a far frame straight to
    a single fixed reference gives few/weak correspondences, which RANSAC
    can turn into a degenerate homography (confirmed -- produced a mirrored,
    kaleidoscopic warp). Adjacent-frame motion is small, so each step is
    well-constrained; remove_objects() composes these into short local
    chains rather than one long global one."""
    orb = cv2.ORB_create(3000)
    kps, descs = zip(*(orb.detectAndCompute(g, None) for g in grays))
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ident = np.eye(3, dtype=np.float32)

    def step_h(a, b):
        da, db = descs[a], descs[b]
        if da is None or db is None or len(da) < 8 or len(db) < 8:
            return ident
        matches = bf.match(da, db)
        if len(matches) < 8:
            return ident
        src = np.float32([kps[a][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kps[b][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        Hmat, _ = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        return Hmat.astype(np.float32) if Hmat is not None else ident

    return [step_h(i, i + 1) for i in range(len(grays) - 1)]

def _local_plate(frames, fwd, t, W, H, window, stride):
    """Background plate for frame t alone, built from a sparse +/-window
    of neighbors (every `stride`-th frame), aligned into frame t's own
    coordinate system by composing adjacent-frame steps.

    Two competing failure modes drove both parameters. Chain length ->
    drift: composing all the way from one global reference accumulates
    enough error over a long clip -- confirmed on a 3s/77-frame tracking
    shot -- that content with real height above the ground plane (hedges,
    tree canopies), which a flat-ground homography can't fully compensate
    for, shows up as large false-positive diffs, sometimes as an outright
    degenerate/mirrored warp for frames far from the reference. So: bound
    the chain with a local window. But too short a window has the opposite
    failure -- confirmed with window=12 (~0.5s at 25fps): a tightly tracked
    object barely moves in-frame over that short a span, so the window
    never contains a clean look at the background under it, and removal
    silently fails again, just for a different reason. Fix: keep the
    stacked frame count roughly fixed (bounding both compute and worst-case
    chain length per stacked frame) but space samples out with `stride`, so
    the window covers several times the temporal reach for the same cost --
    giving the object time to actually move without lengthening any single
    homography chain by more than `stride` steps per hop."""
    T = len(frames)
    idx = sorted(set([t] + list(range(t, max(-1, t - window - 1), -stride))
                      + list(range(t, min(T, t + window + 1), stride))))
    idx = [i for i in idx if 0 <= i < T]
    ident = np.eye(3, dtype=np.float32)
    ones = np.full((H, W), 255, np.uint8)
    stack = np.empty((len(idx), H, W, 3), np.float32)
    vmask = np.empty((len(idx), H, W), bool)

    def step_span(a, b):
        """Homography mapping frame a -> frame b, composed over |a-b| steps."""
        Hm = ident
        if a < b:
            for k in range(a, b): Hm = fwd[k] @ Hm
        else:
            for k in range(a - 1, b - 1, -1): Hm = np.linalg.inv(fwd[k]) @ Hm
        return Hm

    for n, i in enumerate(idx):
        Hm = step_span(t, i)
        stack[n] = cv2.warpPerspective(frames[i], Hm, (W, H), flags=cv2.INTER_LINEAR)
        vmask[n] = cv2.warpPerspective(ones, Hm, (W, H), flags=cv2.INTER_NEAREST) > 0
    stack[~vmask] = np.nan
    with np.errstate(invalid="ignore"):
        plate = np.nanmedian(stack, axis=0)
    unseen = np.isnan(plate)
    if unseen.any():
        plate[unseen] = frames[t].astype(np.float32)[unseen]

    # Confidence = fraction of the window's samples that actually agree with
    # the median at each pixel. A tightly tracked object doesn't fully clear
    # any pixel across the whole window -- it's present in, say, 40% of the
    # aligned samples there rather than 0% or 100% -- so the median still
    # "wins" a value, but neighboring pixels can pick opposite winners from
    # sample to sample, which composited into the frame reads as a smeared,
    # multi-car ghost trail rather than a clean reveal or an honest miss.
    # Low agreement is the signature of exactly that contamination.
    with np.errstate(invalid="ignore"):
        devmax = np.nanmax(np.abs(stack - plate[None]), axis=3)
    valid = ~np.isnan(devmax)
    valid_count = valid.sum(0)
    agree_count = np.where(valid, devmax < 20, False).sum(0)
    confidence = np.divide(agree_count, valid_count, out=np.zeros((H, W), np.float32),
                           where=valid_count > 0)
    return np.clip(plate, 0, 255).astype(np.uint8), confidence

def remove_objects(frames, target=None, window=36, stride=3):
    """Remove moving objects via a homography-aligned local background plate.

    A plain per-frame median (diff each frame against the stack's pixelwise
    median) only recovers background where the object doesn't dominate a
    pixel's temporal history -- true for a hovering camera, but it silently
    does nothing on a tracking shot, where the camera follows the object so
    it sits at ~the same screen position in every frame and the median IS
    the object. Fix: estimate short-range camera ego-motion between
    consecutive frames from background feature matches (RANSAC-robust to
    the object, which is an outlier), and for each output frame build its
    own background from a sparse +/-window of aligned neighbors (see
    _local_plate for why it's sparse, not just short). In that aligned space
    the object -- which moves through the world independently of the camera
    -- lands at a different position in most of the window, so the median
    recovers real background even though it never does in any single raw
    frame. Degrades gracefully to the old behavior when the camera is
    genuinely static (homographies ~ identity).

    `target`: the noun phrase from the instruction (e.g. "black car"). The
    motion detector only ever proposes candidates by shape (compact, solid,
    moving) -- it can't tell a car from a person, or one car from another.
    When `target` names something specific (not the generic "objects"
    fallback), each candidate is also checked against it via CLIP
    (vlm_match.py) before being removed, so "the black car" doesn't also
    take out every other vehicle in frame."""
    H, W = frames[0].shape[:2]; T = len(frames)
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    fwd = _estimate_step_homographies(grays)

    verify = None
    if target and target != "objects":
        from vlm_match import matches_target
        verify = lambda crop: matches_target(crop, target)

    out = []
    for t in range(T):
        bg_t, conf_t = _local_plate(frames, fwd, t, W, H, window, stride)
        mask = _mask_from_diff(frames[t], bg_t, verify=verify)
        mf = cv2.GaussianBlur(mask, (0, 0), 1.5).astype(np.float32)[..., None] / 255.0
        # Gate the blend by per-pixel confidence: where the window's samples
        # didn't actually agree (still-contaminated median, see
        # _local_plate), fade the replacement back toward the original
        # pixel instead of painting in a low-confidence guess -- an honest
        # "didn't remove it here" beats a smeared ghost.
        mf = mf * cv2.GaussianBlur(conf_t, (0, 0), 1.5)[..., None]
        out.append((frames[t]*(1-mf) + bg_t*mf).astype(np.uint8))
    return out, out[T // 2]
