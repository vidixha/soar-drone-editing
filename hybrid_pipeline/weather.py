"""Weather effects for the AERIE aerial video editor. Training-free, all
depth-driven -- ported from the project's original soar_router/modules.py
(pre-dating this branch) back into the pipeline. Excluded from this branch
at first ("do not commit weather for now") while the router/removal/
trajectory work was being stabilized; reintroduced once that was solid.

Each function takes a list of BGR frames + depth (see modules._depth_at:
either one (H,W) map reused every frame, or a (T,H,W) per-frame sequence
from Trace Anything) and returns edited frames. No model, no training --
physically-motivated compositing (Beer-Lambert transmission for fog/haze,
a depth-derived collision surface for rain/snow particles) using whatever
depth backend the router already fetched for this clip.

Real per-frame depth matters more here than for removal: a static depth
map reused across every frame drifts out of alignment with the scene on
any clip with real camera motion (exactly the tracking-shot case this
branch spent most of its effort generalizing removal to) -- collision
surfaces and transmission falloff would silently use the wrong geometry.
Pass a (T,H,W) sequence (--use-trace-anything) for anything but a
near-static hover shot.
"""
import cv2
import numpy as np

from modules import _depth_at

FOG_BETA = {"light": 1.3, "medium": 2.1, "heavy": 3.4}   # density by intensity word

def apply_fog(frames, depth, intensity="medium", gamma=1.3, A=(206, 210, 214)):
    """depth: (H,W) single map (reused every frame) or (T,H,W) per-frame
    sequence (Trace Anything) -- see _depth_at(). Per-frame depth matters
    here specifically for any clip with real camera motion; a static map
    would drift out of alignment with the scene as the view changes."""
    beta = FOG_BETA.get(intensity, 2.1)
    A = np.array(A, np.float32)
    out = []
    for i, f in enumerate(frames):
        d = _depth_at(depth, i)
        dist = np.clip(1 - d, 0, 1)
        t = np.exp(-beta * (dist ** gamma))[..., None]
        blur = cv2.GaussianBlur(f, (0, 0), 1.6).astype(np.float32)
        base = f.astype(np.float32) * t + blur * (1 - t)
        out.append(np.clip(base * t + A * (1 - t), 0, 255).astype(np.uint8))
    return out

# ------------------------------------------------------ collision (Let-It-Snow port)
# Ported from "Let It Snow" (Fiebelman et al.): their MPM physics gives particles
# real gravity/wind/collision against scene geometry. We port that MOTION idea
# training-free using our monocular depth as a collision surface -- NOT their
# Score-Distillation-Sampling appearance stage, which needs per-scene diffusion
# optimization (training) and multi-view capture (3DGS) we don't have. That stage
# is also specifically where their wet-road look comes from, so surface wetness
# stays an open limitation (consistent with the SDXL/ControlNet tests).
def collision_skyline(depth, horizon=None, thresh=0.085, top_margin=None):
    """Per-column collision surface: topmost 'solid' row (treeline/roofline
    silhouette against sky) unioned with the ground horizon. Falling particles
    stop here instead of passing through scene geometry.

    top_margin excludes the very top rows from the scan: after a viewpoint
    change (e.g. TrajectoryCrafter reprojection), the topmost strip can carry
    warp-boundary artifacts with noisy/degenerate depth, which otherwise gets
    misread as a near surface -> spurious instant "landings" at row 0.

    Known limitation, not yet fixed: this treats any "solid" pixel uniformly,
    with no separation between static background geometry and a moving
    foreground object -- a real 4D reconstruction (depth + explicit
    static/dynamic segmentation, tracked over time, e.g. Vista4D's approach)
    would give a stabler, less frame-to-frame-jittery collision surface
    specifically around moving objects, where per-frame monocular depth is
    noisiest. Not yet built; current version is the single-frame-depth
    version validated in this project's earlier work."""
    H, W = depth.shape; horizon = horizon or int(H * 0.55)
    top_margin = top_margin if top_margin is not None else max(6, int(H * 0.02))
    solid = (depth > thresh).astype(np.uint8)
    sky = np.full(W, horizon, np.int32)
    for x in range(W):
        col = np.where(solid[top_margin:horizon, x] > 0)[0]
        if len(col): sky[x] = col[0] + top_margin
    return cv2.GaussianBlur(sky.astype(np.float32).reshape(1, -1), (0, 0), 1.5).flatten().astype(np.int32)

def apply_rain(frames, depth=None, seed=3, n=260, gravity=2.6, wind=0.9):
    """depth: (H,W) single map or (T,H,W) per-frame sequence. Collision
    skyline is recomputed per-frame when a sequence is given (real geometry
    for that frame), otherwise computed once (cheap path, matches a static
    camera+scene)."""
    H, W = frames[0].shape[:2]
    per_frame = depth is not None and depth.ndim == 3
    skyline = None if per_frame else (collision_skyline(depth) if depth is not None else np.full(W, int(H * 0.55), np.int32))
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, W, n); y = rng.uniform(-H, 0, n); vy = rng.uniform(14, 20, n); vx = np.full(n, wind)
    splash_t = np.zeros(n, np.int32); splash_xy = np.zeros((n, 2), np.float32)
    def atmos(f):
        img = f.astype(np.float32); img = (img - 128) * 0.82 + 128; img = img * 0.80 + 18
        hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= 0.6; img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) + np.array([16, 6, -6])
        yy = np.clip(np.linspace(1.2, 0, H), 0, 1)[:, None, None]
        return np.clip(img * (1 - 0.35 * yy) + np.array([200, 198, 196]) * (0.35 * yy), 0, 255)
    out = []
    for k in range(len(frames)):
        if per_frame: skyline = collision_skyline(depth[min(k, depth.shape[0] - 1)])
        lay = np.zeros((H, W), np.float32); splashlay = np.zeros((H, W), np.float32)
        vy2 = vy + gravity; x2 = x + vx; y2 = y + vy2
        for i in range(n):
            xi = int(np.clip(x[i], 0, W - 1)); surf = skyline[xi]
            hit = y2[i] >= surf and y[i] < surf
            yy2 = min(y2[i], surf) if hit else y2[i]
            if 0 <= y[i] < H or 0 <= yy2 < H:
                cv2.line(lay, (int(x[i]), int(np.clip(y[i], 0, H))), (int(x2[i]), int(np.clip(yy2, 0, H))), 0.6, 1, cv2.LINE_AA)
            if hit:
                splash_t[i] = 3; splash_xy[i] = [x[i], surf]
                x[i] = rng.uniform(0, W); y[i] = rng.uniform(-40, -5); vy[i] = rng.uniform(14, 20)
            else:
                x[i], y[i] = x2[i] % W, y2[i]; vy[i] = vy2[i]
            if splash_t[i] > 0:
                cv2.circle(splashlay, (int(splash_xy[i, 0]), int(splash_xy[i, 1])), 4 - splash_t[i], 0.5, 1, cv2.LINE_AA)
                splash_t[i] -= 1
        lay = cv2.GaussianBlur(lay, (0, 0), 0.5)
        r = np.clip(lay, 0, 1)[..., None]; s = np.clip(splashlay, 0, 1)[..., None]; a = atmos(frames[k])
        a = a * (1 - r) + np.array([225, 225, 230]) * r; a = a * (1 - s) + np.array([235, 238, 240]) * s
        out.append(np.clip(a, 0, 255).astype(np.uint8))
    return out

# (accum strength, haze beta, wash strength, particle-count scale) -- wash
# scales the base contrast-cut/white-blend/desaturation/cool-cast, which used
# to be constant regardless of intensity. haze_beta drives an exponential
# transmission falloff (t=exp(-beta*dist)); medium/heavy were left at their
# original un-tuned values (beta=2.4/3.2 -> ~91%/~96% of far regions blended
# to white -- a near-total whiteout, exactly the "too much" complaint) even
# after light was fixed. Retuned so far-region haze forms a real light<medium
# <heavy progression instead of "barely there" then "solid whiteout": light
# ~11%, medium ~35%, heavy ~65% blend at the farthest points.
SNOW_P = {"light": (0.08, 0.12, 0.15, 0.4), "medium": (0.25, 0.43, 0.4, 0.7), "heavy": (0.5, 1.05, 0.7, 1.0)}

def apply_snow(frames, depth, intensity="medium", seed=7, n=260, gravity=0.045, wind=0.35):
    """depth: (H,W) single map or (T,H,W) per-frame sequence -- collision
    skyline, ground-accumulation weighting, and haze all use the current
    frame's own depth when a sequence is given."""
    acc_str, haze_beta, wash, pscale = SNOW_P.get(intensity, SNOW_P["medium"])
    n = max(20, int(n * pscale))
    H, W = frames[0].shape[:2]
    per_frame = depth.ndim == 3
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, W, n); y = rng.uniform(-H, 0, n); vy = rng.uniform(0.6, 1.2, n)
    vx = rng.uniform(-wind, wind, n); ph = rng.uniform(0, 2 * np.pi, n)
    accum = np.zeros((H, W), np.float32)
    b, g, r = [frames[0][..., i].astype(np.int16) for i in range(3)]
    grass = ((g > r + 4) & (g > b - 10)).astype(np.float32)
    def atmos(f, acc, d):
        img = f.astype(np.float32)
        img = (img - 128) * (1 - 0.15 * wash) + 128
        img = img * (1 - 0.10 * wash) + np.array([245, 245, 248]) * (0.10 * wash)
        hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= (1 - 0.5 * wash)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) + wash * np.array([10, 3, -3])
        ground_acc = cv2.GaussianBlur(grass * np.clip((d - 0.35) / 0.5, 0, 1), (0, 0), 3)
        w = np.clip(ground_acc * 0.65 * (acc_str / 0.6) + acc, 0, 1)[..., None]
        img = img * (1 - w) + np.array([248, 248, 250]) * w
        t = np.exp(-haze_beta * np.clip(1 - d, 0, 1))[..., None]; A = np.array([240, 242, 245], np.float32)
        return np.clip(img * t + A * (1 - t), 0, 255)
    out = []
    skyline = None if per_frame else collision_skyline(depth)
    for k in range(len(frames)):
        d = depth[min(k, depth.shape[0] - 1)] if per_frame else depth
        if per_frame: skyline = collision_skyline(d)
        vx_t = vx + 0.15 * np.sin(0.05 * k + ph); x2 = x + vx_t; y2 = y + vy
        lay = np.zeros((H, W), np.float32)
        for i in range(n):
            xi = int(np.clip(x[i], 0, W - 1)); surf = skyline[xi]
            hit = y2[i] >= surf and y[i] < surf
            if hit:
                accum[max(0, int(surf) - 1):int(surf) + 2, max(0, xi - 2):xi + 3] += 0.14
                x[i] = rng.uniform(0, W); y[i] = rng.uniform(-40, -5); vy[i] = rng.uniform(0.6, 1.2); vx[i] = rng.uniform(-wind, wind)
            else:
                x[i], y[i] = x2[i] % W, y2[i]
                # only draw once the flake has actually entered the frame --
                # clamping negative y (not-yet-visible flakes) to row 0 was
                # forcing every off-screen flake onto the top edge, producing
                # a dense comb line there (mistaken for a collision-depth bug
                # earlier; it's purely this draw-clamp).
                if 0 <= y[i] < H:
                    cv2.circle(lay, (int(x[i]), int(y[i])), 2, 0.85, -1, cv2.LINE_AA)
        accum = np.clip(accum, 0, 0.9)
        base = atmos(frames[k], cv2.GaussianBlur(accum, (0, 0), 1.2), d)
        lay = cv2.GaussianBlur(lay, (0, 0), 0.6)[..., None]
        out.append(np.clip(base * (1 - lay) + np.array([252, 252, 255]) * lay, 0, 255).astype(np.uint8))
    return out

def apply_sandstorm(frames, depth, seed=11, n=500, wind=34):
    """Wind-driven dust: mostly-horizontal particles + warm depth-graded haze.
    Additive/atmospheric like fog/snow -- not a reflectance change -- so, unlike
    wet rain, this ports cleanly with no training-free blocker."""
    H, W = frames[0].shape[:2]
    rng = np.random.default_rng(seed)
    x = rng.uniform(-40, W + 40, n); y = rng.uniform(0, H, n); vy = rng.uniform(-1.2, 1.2, n)
    A = np.array([176, 196, 214], np.float32)
    def atmos(f, d):
        img = f.astype(np.float32)
        hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= 0.55; img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        img = img * 0.8 + A * 0.2
        t = np.exp(-3.0 * np.clip(1 - d, 0, 1) ** 1.1)[..., None]
        return np.clip(img * t + A * (1 - t), 0, 255)
    out = []
    for k in range(len(frames)):
        base = atmos(frames[k], _depth_at(depth, k)); lay = np.zeros((H, W), np.float32)
        x2 = x + wind; y2 = y + vy
        for i in range(n): cv2.line(lay, (int(x[i]), int(y[i])), (int(x2[i]), int(y2[i])), 0.22, 2, cv2.LINE_AA)
        x, y = x2, y2; wrap = x > W + 40; x[wrap] -= (W + 80)
        lay = cv2.GaussianBlur(lay, (0, 0), sigmaX=3.5, sigmaY=0.4)
        r = np.clip(lay, 0, 1)[..., None]
        out.append(np.clip(base * (1 - r) + A * r, 0, 255).astype(np.uint8))
    return out

WEATHER = {"fog": apply_fog, "rain": apply_rain, "snow": apply_snow, "sandstorm": apply_sandstorm}
