"""Metric, geometry-aware weather over reconstructed source frames."""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .config import SnowConfig
from .warp_particles import (
    WarpFlakeEngine,
    available as warp_available,
    build_depth_mesh,
    seed_settled_on_mesh,
)


class SnowResult(NamedTuple):
    video: np.ndarray
    weather_mask: np.ndarray
    final_positions: np.ndarray
    landed: np.ndarray
    emitted: int
    collision_count: int
    surface_points: int
    collision_radius: float
    frustum_volume: float
    target_particles: int
    warmup_frames: int
    emission_rate: int


SNOW_COLOUR = np.array([242, 242, 245], dtype=np.float32)
RAIN_COLOUR = np.array([176, 186, 198], dtype=np.float32)
SAND_COLOUR = np.array([214, 186, 142], dtype=np.float32)
FOG_COLOUR = np.array([206, 210, 214], dtype=np.float32)
RAIN_GRADE = np.array([0.86, 0.88, 0.93], dtype=np.float32)
SNOW_SKY_ZENITH = np.array([156.0, 162.0, 172.0], dtype=np.float32)
SNOW_SKY_HORIZON = np.array([214.0, 218.0, 222.0], dtype=np.float32)
RAIN_SKY_ZENITH = np.array([42.0, 48.0, 58.0], dtype=np.float32)
RAIN_SKY_HORIZON = np.array([96.0, 104.0, 112.0], dtype=np.float32)
SAND_SKY_ZENITH = np.array([118.0, 96.0, 64.0], dtype=np.float32)
SAND_SKY_HORIZON = np.array([196.0, 168.0, 118.0], dtype=np.float32)
FOG_SKY_ZENITH = np.array([168.0, 172.0, 178.0], dtype=np.float32)
FOG_SKY_HORIZON = np.array([206.0, 210.0, 214.0], dtype=np.float32)
SNOW_FOG_COLOUR = SNOW_SKY_HORIZON.copy()
RAIN_FOG_COLOUR = RAIN_SKY_HORIZON.copy()
SAND_FOG_COLOUR = SAND_SKY_HORIZON.copy()
FOG_FOG_COLOUR = FOG_SKY_HORIZON.copy()
MESH_EFFECTS = {"snow", "sandstorm"}


def _sky_palette(effect: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if effect == "rain":
        return RAIN_SKY_ZENITH, RAIN_SKY_HORIZON, RAIN_FOG_COLOUR, 7.0
    if effect == "sandstorm":
        return SAND_SKY_ZENITH, SAND_SKY_HORIZON, SAND_FOG_COLOUR, 9.0
    if effect == "fog":
        return FOG_SKY_ZENITH, FOG_SKY_HORIZON, FOG_FOG_COLOUR, 3.5
    return SNOW_SKY_ZENITH, SNOW_SKY_HORIZON, SNOW_FOG_COLOUR, 5.0


def _particle_colour(effect: str) -> np.ndarray:
    if effect == "rain":
        return RAIN_COLOUR
    if effect == "sandstorm":
        return SAND_COLOUR
    if effect == "fog":
        return FOG_COLOUR
    return SNOW_COLOUR


def _normalise(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else fallback.astype(np.float32)


def world_up(cam_c2w: np.ndarray) -> np.ndarray:
    """Estimate world up from OpenCV camera down axes."""
    return _normalise(
        np.median(-cam_c2w[:, :3, 1], axis=0).astype(np.float32),
        np.array([0.0, 1.0, 0.0]),
    )


def _scene_basis(cam_c2w: np.ndarray, up: np.ndarray) -> np.ndarray:
    right = cam_c2w[0, :3, 0].astype(np.float32)
    right -= up * float(np.dot(right, up))
    right = _normalise(right, np.array([1.0, 0.0, 0.0]))
    forward = cam_c2w[0, :3, 2].astype(np.float32)
    forward -= up * float(np.dot(forward, up))
    forward -= right * float(np.dot(forward, right))
    forward = _normalise(forward, np.cross(right, up))
    return np.stack((right, up, forward), axis=1)


def _unproject(
    depth: np.ndarray,
    camera: np.ndarray,
    intrinsics: np.ndarray,
    mask: np.ndarray,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    ys, xs = np.arange(0, height, stride), np.arange(0, width, stride)
    grid_x, grid_y = np.meshgrid(xs, ys)
    sampled = depth[np.ix_(ys, xs)].astype(np.float32)
    valid = mask[np.ix_(ys, xs)] & np.isfinite(sampled) & (sampled > 0) & (sampled < 999)
    fx, fy, cx, cy = intrinsics.astype(np.float32)
    camera_points = np.stack(
        ((grid_x - cx) / fx * sampled, (grid_y - cy) / fy * sampled, sampled),
        axis=-1,
    )
    world = camera_points @ camera[:3, :3].astype(np.float32).T + camera[:3, 3]
    dx = np.roll(world, -1, axis=1) - np.roll(world, 1, axis=1)
    dy = np.roll(world, -1, axis=0) - np.roll(world, 1, axis=0)
    normals = np.cross(dx, dy)
    lengths = np.linalg.norm(normals, axis=-1)
    neighbours = (
        valid
        & np.roll(valid, -1, axis=0)
        & np.roll(valid, 1, axis=0)
        & np.roll(valid, -1, axis=1)
        & np.roll(valid, 1, axis=1)
    )
    neighbours[[0, -1], :] = False
    neighbours[:, [0, -1]] = False
    keep = neighbours & (lengths > 1e-8)
    normals[keep] /= lengths[keep, None]
    return world[keep].astype(np.float32), normals[keep].astype(np.float32)


def build_surface_cloud(
    depths: np.ndarray,
    cam_c2w: np.ndarray,
    intrinsics: np.ndarray,
    static_masks: np.ndarray,
    *,
    stride: int,
    frame_stride: int,
    up: np.ndarray,
    sky_masks: np.ndarray | None = None,
    voxel_scale: float = 0.003,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse sampled reconstructed depths into a compact collision cloud.

    Sky pixels are never unprojected, so they cannot be back-projected later.
    """
    point_batches, normal_batches = [], []
    for index in range(0, len(depths), max(1, frame_stride)):
        usable = static_masks[index]
        if sky_masks is not None:
            usable = usable & ~sky_masks[index]
        points, normals = _unproject(
            depths[index], cam_c2w[index], intrinsics[index], usable, max(2, stride)
        )
        if len(points):
            point_batches.append(points)
            normal_batches.append(normals)
    if not point_batches:
        raise ValueError("reconstruction contains no valid static depth samples")
    points = np.concatenate(point_batches)
    normals = np.concatenate(normal_batches)
    normals[(normals @ up) < 0] *= -1
    low, high = np.percentile(points, (1, 99), axis=0)
    voxel = max(float(np.linalg.norm(high - low)) * voxel_scale, 1e-5)
    _, representatives = np.unique(
        np.floor(points / voxel).astype(np.int64), axis=0, return_index=True
    )
    return points[representatives], normals[representatives]


def _project(
    points: np.ndarray, camera: np.ndarray, intrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_points = (points - camera[:3, 3]) @ camera[:3, :3]
    z = camera_points[:, 2]
    fx, fy, cx, cy = intrinsics.astype(np.float32)
    safe_z = np.maximum(z, 1e-6)
    return (
        fx * camera_points[:, 0] / safe_z + cx,
        fy * camera_points[:, 1] / safe_z + cy,
        z,
    )


def _terminal_speed(diameters_mm: np.ndarray, effect: str) -> np.ndarray:
    if effect == "rain":
        return 3.4 + 2.6 * np.power(diameters_mm, 0.5)
    if effect == "sandstorm":
        return 1.15 + 0.42 * np.power(diameters_mm, 0.55)
    if effect == "fog":
        return 0.12 + 0.08 * np.power(diameters_mm, 0.4)
    return 0.55 + 0.38 * np.power(diameters_mm, 0.55)


def _response_seconds(diameters_mm: np.ndarray, effect: str) -> np.ndarray:
    if effect == "rain":
        return np.clip(0.04 + diameters_mm * 0.02, 0.05, 0.18)
    if effect == "sandstorm":
        return np.clip(0.10 + diameters_mm * 0.05, 0.12, 0.45)
    if effect == "fog":
        return np.clip(0.45 + diameters_mm * 0.12, 0.4, 1.2)
    return np.clip(0.18 + diameters_mm * 0.08, 0.2, 0.8)


def _unproject_pixels(
    depth: np.ndarray,
    camera: np.ndarray,
    intrinsics: np.ndarray,
    keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject non-sky pixels into the world. Sky never becomes a 3D point."""
    height, width = depth.shape
    ys, xs = np.indices((height, width), dtype=np.float32)
    fx, fy, cx, cy = intrinsics.astype(np.float32)
    z = depth.astype(np.float32)
    valid = keep & np.isfinite(z) & (z > 0) & (z < 999)
    camera_points = np.stack(((xs - cx) / fx * z, (ys - cy) / fy * z, z), axis=-1)
    world = camera_points @ camera[:3, :3].astype(np.float32).T + camera[:3, 3].astype(
        np.float32
    )
    return world, valid


def _sample_world_field(
    world: np.ndarray,
    valid: np.ndarray,
    points: np.ndarray,
    values: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Read a world-space scalar at each unprojected pixel (IDW, camera-independent)."""
    sampled = np.zeros(world.shape[:2], dtype=np.float32)
    if not valid.any() or not len(points):
        return sampled
    tree = cKDTree(points)
    neighbours = min(4, len(points))
    distance, index = tree.query(
        world[valid],
        k=neighbours,
        distance_upper_bound=max(radius, 1e-4),
        workers=-1,
    )
    if neighbours == 1:
        distance = distance[:, None]
        index = index[:, None]
    distance = np.maximum(distance.astype(np.float32), 1e-4)
    hit = np.isfinite(distance) & (index < len(points))
    weight = np.where(hit, 1.0 / (distance * distance), 0.0)
    total = weight.sum(axis=1)
    field = np.where(hit, values[np.clip(index, 0, max(len(values) - 1, 0))], 0.0)
    blended = np.divide(np.sum(weight * field, axis=1), total, out=np.zeros(len(total)), where=total > 0)
    sampled[valid] = np.clip(blended, 0, 1)
    return sampled


def _project_static_cover(
    points: np.ndarray,
    camera: np.ndarray,
    intrinsics: np.ndarray,
    height: int,
    width: int,
    radius_metres: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project the fused cloud only. No per-frame depth, so no depth flicker."""
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    if not len(points):
        return np.zeros((height, width), dtype=bool), zbuf
    x, y, z = _project(points, camera, intrinsics)
    px, py = np.rint(x).astype(np.int32), np.rint(y).astype(np.int32)
    visible = (z > 1e-4) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
    if not visible.any():
        return np.zeros((height, width), dtype=bool), zbuf
    focal = float(np.mean(intrinsics[:2]))
    radii = np.clip(
        np.rint(focal * max(radius_metres, 1e-4) / np.maximum(z[visible], 1e-3)),
        1,
        6,
    ).astype(np.int32)
    px, py, z = px[visible], py[visible], z[visible].astype(np.float32)
    for radius in np.unique(radii):
        select = radii == radius
        layer = np.full((height, width), np.inf, dtype=np.float32)
        np.minimum.at(layer, (py[select], px[select]), z[select])
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(2 * radius + 1), int(2 * radius + 1))
        )
        stamped = np.where(np.isfinite(layer), -layer, -1.0e9)
        stamped = cv2.dilate(stamped, kernel)
        layer = np.where(stamped > -1.0e8, -stamped, np.inf)
        zbuf = np.minimum(zbuf, layer)
    close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cover = cv2.morphologyEx(
        np.isfinite(zbuf).astype(np.uint8), cv2.MORPH_CLOSE, close
    ).astype(bool)
    filled = np.where(np.isfinite(zbuf), -zbuf, -1.0e9)
    filled = cv2.dilate(filled, close)
    zbuf = np.where(cover, -filled, np.inf)
    return cover, zbuf


def _blue_sky_pixels(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    sat, value = hsv[:, :, 1], hsv[:, :, 2]
    red, green, blue = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    return (blue > red + 15) & (blue > green + 4) & (value > 120) & (sat > 28)


def _image_sky_mask(frame: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Sky for this camera frame. White buildings are never sky.

    Blue pixels only. Far/invalid depth is not enough — AM marks silos as
    invalid on some frames, and that was eating buildings.
    """
    color = _blue_sky_pixels(frame)
    geometry = np.isfinite(depth) & (depth > 0) & (depth < 800) & ~color
    like = color & ~geometry
    like = cv2.morphologyEx(like.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    like[geometry] = 0
    _count, labels = cv2.connectedComponents(like, 8)
    keep = np.zeros(frame.shape[:2], dtype=bool)
    for label in np.unique(labels[0]):
        if label != 0:
            keep |= labels == label
    keep[geometry] = False
    if min(frame.shape[:2]) > 64:
        keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        keep[geometry] = 0
    return keep.astype(bool)


def _sky_homography(source_camera: np.ndarray, dest_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Map sky pixels between views. Sky is at infinity, so rotation only."""
    fx, fy, cx, cy = intrinsics.astype(np.float64)
    kmat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return kmat @ dest_camera[:3, :3].T @ source_camera[:3, :3] @ np.linalg.inv(kmat)


def _temporal_or(masks: np.ndarray, radius: int = 2) -> np.ndarray:
    """Screen-space OR. Locks geometry that AM depth only sees on some frames."""
    frames = len(masks)
    if frames <= 1 or radius <= 0:
        return masks
    padded = np.pad(masks.astype(np.uint8), ((radius, radius), (0, 0), (0, 0)), mode="edge")
    window = np.stack([padded[offset : offset + frames] for offset in range(2 * radius + 1)])
    return window.max(axis=0).astype(bool)


def stabilize_sky_masks(
    masks: np.ndarray, cameras: np.ndarray, intrinsics: np.ndarray, radius: int = 2
) -> np.ndarray:
    """Redo-per-frame masks, then median in the current camera (warped by pose)."""
    frames, height, width = masks.shape
    stable = np.empty(masks.shape, dtype=np.float32)
    for index in range(frames):
        stack = [masks[index].astype(np.float32)]
        for other in range(max(0, index - radius), min(frames, index + radius + 1)):
            if other == index:
                continue
            homography = _sky_homography(cameras[other], cameras[index], intrinsics[index])
            stack.append(
                cv2.warpPerspective(
                    masks[other].astype(np.float32),
                    homography,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            )
        aligned = np.median(np.stack(stack), axis=0)
        # Conservative: sky only where this frame AND the pose-aligned window agree.
        # Stops buildings popping in/out when one frame floods onto a silo.
        stable[index] = masks[index].astype(np.float32) * aligned
    return stable


def complete_sky(frame: np.ndarray, sky_mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Grow the sky matte down to the real horizon without eating buildings.

    Stored sky often stops short because AerialMetric gives pale sky a fake
    depth. Extend each column through low-gradient blue/pale pixels, then keep
    only components that still touch the top of the frame.
    """
    color = _blue_sky_pixels(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    pale = (hsv[:, :, 2] > 145) & (hsv[:, :, 1] < 75)
    far = ~np.isfinite(depth) | (depth <= 0) | (depth >= 999)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gradient = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    height, width = sky_mask.shape
    keep = sky_mask.copy()
    slack = 48
    for column in range(width):
        rows = np.flatnonzero(sky_mask[:, column])
        if not len(rows):
            continue
        keep[rows[0] : rows[-1] + 1, column] = True
        for row in range(rows[-1] + 1, min(height, rows[-1] + 1 + slack)):
            if gradient[row, column] >= 18.0:
                break
            if color[row, column] or pale[row, column] or far[row, column]:
                keep[row, column] = True
            else:
                break
    _count, labels = cv2.connectedComponents(keep.astype(np.uint8), 8)
    output = np.zeros_like(keep)
    for label in np.unique(labels[0]):
        if label != 0:
            output |= labels == label
    return output


def _temporal_majority(masks: np.ndarray, radius: int = 2) -> np.ndarray:
    """Kill single-frame sky sparkle. Screen-space; small radius only."""
    frames = len(masks)
    if frames <= 1 or radius <= 0:
        return masks
    padded = np.pad(masks.astype(np.uint8), ((radius, radius), (0, 0), (0, 0)), mode="edge")
    window = np.stack([padded[offset : offset + frames] for offset in range(2 * radius + 1)])
    return window.mean(axis=0) >= 0.5


def _feather_mask(mask: np.ndarray, pixels: float) -> np.ndarray:
    """Soft sky alpha inside the mask so the horizon does not sparkle."""
    if pixels <= 0:
        return mask.astype(np.float32)
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(distance / pixels, 0, 1).astype(np.float32)


def _view_heading(camera: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = camera[:3, 2].astype(np.float32)
    forward = forward - up * float(np.dot(forward, up))
    return _normalise(forward, np.array([0.0, 0.0, 1.0], dtype=np.float32))


def _sky_shift_pixels(
    camera: np.ndarray,
    reference_heading: np.ndarray,
    up: np.ndarray,
    focal: float,
) -> float:
    heading = _view_heading(camera, up)
    sine = float(np.dot(np.cross(reference_heading, heading), up))
    cosine = float(np.clip(np.dot(reference_heading, heading), -1.0, 1.0))
    return float(np.arctan2(sine, cosine) * focal)


def _smooth_metric_depth(depth: np.ndarray, sky_mask: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth) & (depth > 0) & (depth < 999) & (~sky_mask)
    if not finite.any():
        return depth.astype(np.float32)
    filled = np.where(finite, depth, float(np.median(depth[finite]))).astype(np.float32)
    return cv2.GaussianBlur(filled, (0, 0), 5.0)


def _fog_visibility(config: SnowConfig) -> float:
    if config.fog_visibility_metres > 0:
        return float(config.fog_visibility_metres)
    if config.effect == "fog":
        return 32.0
    if config.effect == "sandstorm":
        return 70.0
    return 160.0 if config.effect == "rain" else 220.0


def _weather_sky(
    height: int,
    width: int,
    *,
    effect: str,
    seed: int,
    shift_x: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    zenith, horizon, fog, noise_amp = _sky_palette(effect)
    lift = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None] ** 0.82
    sky = zenith + (horizon - zenith) * lift[..., None]
    rng = np.random.default_rng(seed * 4099)
    pad = max(64, width // 4)
    coarse = rng.normal(0.0, 1.0, (max(6, height // 22), max(6, (width + 2 * pad) // 22))).astype(
        np.float32
    )
    noise = cv2.GaussianBlur(
        cv2.resize(coarse, (width + 2 * pad, height), interpolation=cv2.INTER_CUBIC),
        (0, 0),
        18.0,
    )
    origin = pad + int(round(shift_x))
    origin = int(np.clip(origin, 0, 2 * pad))
    sky = sky + noise[:, origin : origin + width, None] * noise_amp
    return np.clip(sky, 0, 255), fog


def _sky_box_sample(
    height: int,
    width: int,
    camera: np.ndarray,
    intrinsics: np.ndarray,
    up: np.ndarray,
    zenith: np.ndarray,
    horizon: np.ndarray,
) -> np.ndarray:
    """Look up the weather sky box along each camera ray."""
    ys, xs = np.indices((height, width), dtype=np.float32)
    fx, fy, cx, cy = intrinsics.astype(np.float32)
    rays = np.stack(((xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs)), axis=-1)
    rays = rays / np.clip(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-6, None)
    world = rays @ camera[:3, :3].astype(np.float32).T
    lift = np.clip(world @ up.astype(np.float32), 0, 1) ** 0.72
    return zenith + (horizon - zenith) * (1.0 - lift)[..., None]


def _relight_from_skybox(
    frame: np.ndarray,
    depth: np.ndarray,
    cover: np.ndarray,
    camera: np.ndarray,
    intrinsics: np.ndarray,
    up: np.ndarray,
    sky_lookup: np.ndarray,
    old_sky: np.ndarray,
    config: SnowConfig,
) -> np.ndarray:
    """Relight the plate as if it sits under the new sky box."""
    new_ambient = np.median(sky_lookup.reshape(-1, 3), axis=0)
    old_ambient = np.maximum(old_sky.astype(np.float32), 8.0)
    ratio = np.clip(new_ambient / old_ambient, 0.32, 1.25)
    relit = frame.astype(np.float32) * ratio
    world, valid = _unproject_pixels(depth, camera, intrinsics, cover)
    dx = np.roll(world, -1, axis=1) - np.roll(world, 1, axis=1)
    dy = np.roll(world, -1, axis=0) - np.roll(world, 1, axis=0)
    normals = np.cross(dx, dy)
    lengths = np.linalg.norm(normals, axis=-1)
    keep = valid & (lengths > 1e-8)
    normals[keep] /= lengths[keep, None]
    normals[(normals @ up) < 0] *= -1
    facing = np.clip(normals @ up, 0, 1)
    facing = np.where(keep, facing, 0.0).astype(np.float32)
    facing = cv2.GaussianBlur(facing, (0, 0), 2.0)
    bounce = sky_lookup * (0.18 + 0.28 * facing)[..., None]
    wrap = {"rain": 0.28, "fog": 0.12, "sandstorm": 0.14}.get(config.effect, 0.16)
    relit = relit * (1.0 - wrap * facing[..., None]) + bounce * wrap
    veil = {"rain": 0.20, "fog": 0.35, "sandstorm": 0.22}.get(config.effect, 0.08)
    relit = relit * (1.0 - veil) + sky_lookup * veil
    if config.effect == "rain":
        luma = np.clip(relit.mean(axis=-1) / 255.0, 0, 1)
        spec = np.clip(luma - 0.42, 0, 1) * facing * 0.16
        relit = relit + sky_lookup * spec[..., None]
    return np.clip(relit, 0, 255)


def _composite_cover(frame: np.ndarray, surface: np.ndarray, colour: np.ndarray) -> np.ndarray:
    """Settled flakes follow mesh shading. Not a flat stamp."""
    luma = cv2.GaussianBlur(frame.mean(axis=-1).astype(np.float32) / 255.0, (0, 0), 1.2)
    amount = np.clip(surface, 0, 1)
    tint = colour * (0.70 + 0.30 * luma[..., None])
    tint = 0.82 * tint + 0.18 * frame
    return frame * (1.0 - amount[..., None]) + tint * amount[..., None]


def _composite_over_sky(
    frame: np.ndarray,
    depth: np.ndarray,
    sky_matte: np.ndarray,
    camera: np.ndarray,
    intrinsics: np.ndarray,
    up: np.ndarray,
    config: SnowConfig,
    *,
    sky_shift: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full sky box + relight. Buildings stay locked; leftover blue sky is filled."""
    height, width = frame.shape[:2]
    zenith, horizon, _, _ = _sky_palette(config.effect)
    sky, _fog = _weather_sky(
        height, width, effect=config.effect, seed=config.seed, shift_x=sky_shift
    )
    sky_lookup = _sky_box_sample(height, width, camera, intrinsics, up, zenith, horizon)
    sky = 0.65 * sky + 0.35 * sky_lookup
    sky_hard = sky_matte > 0.45
    cover = ~sky_hard
    zbuf = np.where(
        cover & np.isfinite(depth) & (depth > 0) & (depth < 999),
        depth.astype(np.float32),
        np.inf,
    )
    old_sky = (
        np.median(frame[sky_hard], axis=0).astype(np.float32)
        if sky_hard.any()
        else np.array([165.0, 195.0, 230.0], dtype=np.float32)
    )
    relit = _relight_from_skybox(
        frame, depth, cover, camera, intrinsics, up, sky, old_sky, config
    )
    if config.replace_sky:
        replace = _feather_mask(sky_hard, float(np.clip(height / 240.0, 1.5, 3.5)))
        replace = np.maximum(replace, sky_matte.astype(np.float32) * sky_hard)
        output = relit * (1.0 - replace[..., None]) + sky.astype(np.float32) * replace[..., None]
        alpha = 1.0 - replace
    else:
        alpha = cover.astype(np.float32)
        output = relit
    visibility = _fog_visibility(config)
    metric = np.where(np.isfinite(zbuf), zbuf, visibility * 3.0)
    start = visibility * {"rain": 0.10, "fog": 0.05, "sandstorm": 0.08}.get(config.effect, 0.18)
    fog_cap = {"rain": 0.92, "fog": 0.95, "sandstorm": 0.88}.get(config.effect, 0.72)
    fog = np.clip(1.0 - np.exp(-(np.maximum(metric - start, 0.0) / visibility)), 0, fog_cap)
    fog = cv2.GaussianBlur(fog, (0, 0), 2.5) * np.clip(alpha, 0, 1)
    output = output * (1.0 - fog[..., None]) + sky * fog[..., None]
    return output, zbuf, cover


def _visible_particles(
    positions: np.ndarray,
    camera: np.ndarray,
    intrinsics: np.ndarray,
    zbuf: np.ndarray,
    active: np.ndarray,
    slack: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = zbuf.shape
    x, y, z = _project(positions, camera, intrinsics)
    x = np.nan_to_num(x, nan=-1.0)
    y = np.nan_to_num(y, nan=-1.0)
    px, py = np.rint(x).astype(np.int32), np.rint(y).astype(np.int32)
    visible = (z > 1e-4) & (px >= 0) & (px < width) & (py >= 0) & (py < height) & active
    indices = np.flatnonzero(visible)
    if len(indices):
        scene_z = zbuf[py[indices], px[indices]]
        keep = ~np.isfinite(scene_z) | (z[indices] <= scene_z + slack)
        indices = indices[keep]
    return px, py, z, indices


def _render(
    frame: np.ndarray,
    depth: np.ndarray,
    camera: np.ndarray,
    intrinsics: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    diameters_mm: np.ndarray,
    active: np.ndarray,
    landed: np.ndarray,
    up: np.ndarray,
    surface_points: np.ndarray,
    surface_tree: cKDTree,
    *,
    dt: float,
    config: SnowConfig,
    radius_metres: float,
    accumulation_strength: float,
    sky_shift: float = 0.0,
    sky_matte: np.ndarray | None = None,
) -> np.ndarray:
    if sky_matte is None:
        sky_matte = (depth >= 999).astype(np.float32)
    if config.effect == "rain":
        return _render_rain(
            frame,
            depth,
            camera,
            intrinsics,
            positions,
            velocities,
            diameters_mm,
            active,
            surface_points,
            surface_tree,
            dt=dt,
            config=config,
            radius_metres=radius_metres,
            sky_shift=sky_shift,
            sky_matte=sky_matte,
            up=up,
        )
    frame, zbuf, cover = _composite_over_sky(
        frame,
        depth,
        sky_matte,
        camera,
        intrinsics,
        up,
        config,
        sky_shift=sky_shift,
    )
    height, width = frame.shape[:2]
    px, py, z, indices = _visible_particles(
        positions, camera, intrinsics, zbuf, active, 2.0 * radius_metres
    )

    density_layers = [np.zeros((height, width), dtype=np.float32) for _ in range(3)]
    focal = float(np.mean(intrinsics[:2]))
    for particle in indices:
        diameter = max(
            2.0,
            focal * (diameters_mm[particle] / 1000.0) / max(z[particle], 1e-3),
        )
        opacity = float(np.clip(0.22 + 0.12 * diameter, 0.28, 0.78))
        layer_index = 0 if diameter < 2.6 else 1 if diameter < 4.5 else 2
        density = density_layers[layer_index]
        end = (int(px[particle]), int(py[particle]))
        old_x, old_y, _ = _project(
            positions[particle : particle + 1] - velocities[particle : particle + 1] * dt * 0.85,
            camera,
            intrinsics,
        )
        delta = np.array([old_x[0] - end[0], old_y[0] - end[1]], dtype=np.float32)
        length = float(np.linalg.norm(delta))
        if length > 8.0:
            delta *= 8.0 / length
        start = (
            int(np.clip(round(end[0] + delta[0]), 0, width - 1)),
            int(np.clip(round(end[1] + delta[1]), 0, height - 1)),
        )
        if length > 0.8:
            cv2.line(density, start, end, opacity * 0.35, 1, cv2.LINE_AA)
        radius = max(1, int(round(diameter * 0.45)))
        cv2.circle(density, end, radius, opacity, -1, cv2.LINE_AA)

    density = sum(
        cv2.GaussianBlur(layer, (0, 0), sigma)
        for layer, sigma in zip(density_layers, (0.30, 0.60, 1.0), strict=True)
    )
    surface = np.zeros((height, width), dtype=np.float32)
    if config.accumulation and landed.any():
        world, valid = _unproject_pixels(depth, camera, intrinsics, cover)
        surface = _sample_world_field(
            world,
            valid,
            positions[landed],
            np.ones(int(landed.sum()), dtype=np.float32),
            max(radius_metres * 1.1, 0.22),
        )
    falling = np.clip(density * config.intensity, 0, 0.82)
    colour = _particle_colour(config.effect)
    output = _composite_cover(frame.astype(np.float32), surface, colour)
    output = output * (1.0 - falling[..., None]) + colour * falling[..., None]
    return np.clip(output, 0, 255).astype(np.uint8)


def _render_rain(
    frame: np.ndarray,
    depth: np.ndarray,
    camera: np.ndarray,
    intrinsics: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    diameters_mm: np.ndarray,
    active: np.ndarray,
    surface_points: np.ndarray,
    surface_tree: cKDTree,
    *,
    dt: float,
    config: SnowConfig,
    radius_metres: float,
    sky_shift: float = 0.0,
    sky_matte: np.ndarray | None = None,
    up: np.ndarray | None = None,
) -> np.ndarray:
    height, width = frame.shape[:2]
    if sky_matte is None:
        sky_matte = (depth >= 999).astype(np.float32)
    if up is None:
        up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    frame, zbuf, cover = _composite_over_sky(
        frame,
        depth,
        sky_matte,
        camera,
        intrinsics,
        up,
        config,
        sky_shift=sky_shift,
    )
    px, py, z, indices = _visible_particles(
        positions, camera, intrinsics, zbuf, active, 2.0 * radius_metres
    )

    streaks = np.zeros((height, width), dtype=np.float32)
    focal = float(np.mean(intrinsics[:2]))
    shutter = max(dt * 1.15, 0.028)
    for particle in indices:
        diameter = focal * (diameters_mm[particle] / 1000.0) / max(z[particle], 1e-3)
        opacity = float(np.clip(0.07 + 0.22 * diameter, 0.08, 0.62))
        end = (int(px[particle]), int(py[particle]))
        old_x, old_y, _ = _project(
            positions[particle : particle + 1] - velocities[particle : particle + 1] * shutter,
            camera,
            intrinsics,
        )
        delta = np.array([old_x[0] - end[0], old_y[0] - end[1]], dtype=np.float32)
        length = float(np.linalg.norm(delta))
        max_streak = 36.0
        if length > max_streak:
            delta *= max_streak / length
            length = max_streak
        start = (
            int(np.clip(round(end[0] + delta[0]), 0, width - 1)),
            int(np.clip(round(end[1] + delta[1]), 0, height - 1)),
        )
        thickness = 1 if diameter < 1.8 else 2
        if length < 1.2:
            continue
        cv2.line(streaks, start, end, opacity, thickness, cv2.LINE_AA)

    falling = np.clip(cv2.GaussianBlur(streaks, (0, 0), 0.45) * config.intensity, 0, 0.78)
    output = frame.astype(np.float32)
    output = output * (1 - falling[..., None]) + RAIN_COLOUR * falling[..., None]
    return np.clip(output, 0, 255).astype(np.uint8)


def simulate_snow(
    video: np.ndarray,
    depths: np.ndarray,
    cam_c2w: np.ndarray,
    intrinsics: np.ndarray,
    static_masks: np.ndarray,
    accumulation_masks: np.ndarray | None = None,
    sky_masks: np.ndarray | None = None,
    *,
    fps: float,
    config: SnowConfig,
    object_collision: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> SnowResult:
    """Simulate world-space weather and composite it over reconstructed frames."""
    config.validate()
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"expected RGB video shaped (T,H,W,3), got {video.shape}")
    if depths.shape != video.shape[:3] or static_masks.shape != video.shape[:3]:
        raise ValueError("video, depths, and static masks must have matching dimensions")
    if accumulation_masks is None:
        accumulation_masks = static_masks
    if accumulation_masks.shape != video.shape[:3]:
        raise ValueError("video and accumulation masks must have matching dimensions")
    if sky_masks is None:
        sky_masks = depths >= 999
    if sky_masks.shape != video.shape[:3]:
        raise ValueError("video and sky masks must have matching dimensions")
    if len(cam_c2w) != len(video) or len(intrinsics) != len(video):
        raise ValueError("camera count must match video frame count")
    sky_mattes = sky_masks.astype(np.float32)

    rng = np.random.default_rng(config.seed)
    up = world_up(cam_c2w)
    heading0 = _view_heading(cam_c2w[0], up)
    basis = _scene_basis(cam_c2w, up)
    surface_points, surface_normals = build_surface_cloud(
        depths,
        cam_c2w,
        intrinsics,
        static_masks,
        stride=config.surface_stride,
        frame_stride=config.surface_frame_stride,
        up=up,
        sky_masks=sky_masks,
    )
    origin = np.median(surface_points, axis=0)
    local_surface = (surface_points - origin) @ basis
    lower, upper = np.percentile(local_surface, (1, 99), axis=0)
    span = np.maximum(upper - lower, 1e-3)
    collision_radius = float(np.clip(np.max(span) * 0.0028, 0.45, 2.2))
    radius_metres = max(float(np.linalg.norm(span)) * 0.0022, collision_radius * 0.35)
    accumulation_strength = float(
        np.clip(max(config.accumulation_minutes, 0.0) / 16.0, 0.0, 0.62)
    )
    tree = cKDTree(surface_points)
    if object_collision is not None and len(object_collision) != len(video):
        raise ValueError("object_collision must have one (points, normals) pair per frame")
    valid_depth = depths[
        static_masks & np.isfinite(depths) & (depths > 0) & (depths < 999)
    ]
    far_depth = float(np.percentile(valid_depth, 75))
    near_depth = max(0.5, far_depth * 0.02)
    height, width = depths.shape[1:]
    fx, fy = np.median(intrinsics[:, :2], axis=0)
    frustum_volume = float(
        4
        * (width / (2 * fx))
        * (height / (2 * fy))
        * (far_depth**3 - near_depth**3)
        / 3
    )
    particle_cap = 30_000 if config.effect == "rain" else 12_000 if config.effect == "fog" else 28_000
    raw_count = frustum_volume * config.particles_per_cubic_metre * max(config.intensity, 0.05)
    if config.particles_per_cubic_metre <= 0 and not config.particles_per_frame:
        target_particles = 0
    else:
        target_particles = int(np.clip(raw_count, 64, particle_cap))
    dt = 1 / max(float(fps), 1)
    fall_speed = {"rain": 6.5, "sandstorm": 2.4, "fog": 0.45}.get(config.effect, 1.7)
    lifetime_bounds = (0.4, 3.0) if config.effect == "rain" else (2.0, 12.0) if config.effect == "fog" else (1.0, 8.0)
    lifetime = float(np.clip(span[1] / fall_speed, *lifetime_bounds))
    if target_particles <= 0 and not config.particles_per_frame:
        emission_rate = 0
    else:
        automatic_rate = max(1, round(target_particles / (lifetime * max(float(fps), 1))))
        emission_rate = (
            max(1, round(config.particles_per_frame * max(config.intensity, 0.05)))
            if config.particles_per_frame
            else automatic_rate
        )
    warmup_frames = max(0, round(config.warmup_seconds * max(float(fps), 1)))
    positions = np.empty((0, 3), dtype=np.float32)
    velocities = np.empty((0, 3), dtype=np.float32)
    diameters_mm = np.empty(0, dtype=np.float32)
    active = np.empty(0, dtype=bool)
    landed = np.empty(0, dtype=bool)
    right, _, forward = basis.T
    warp_engine = None
    mesh_verts = np.empty((0, 3), dtype=np.float32)
    mesh_faces = np.empty((0, 3), dtype=np.int32)
    if config.effect in MESH_EFFECTS:
        mesh_verts, mesh_faces = build_depth_mesh(depths, cam_c2w, intrinsics, sky_masks)
        if config.collision and warp_available() and len(mesh_faces) >= 8:
            warp_engine = WarpFlakeEngine(
                mesh_verts,
                mesh_faces,
                up=up,
                right=right,
                forward=forward,
                origin=origin,
                wind=config.wind_metres_per_second,
                turbulence=config.turbulence_metres_per_second,
                max_dist=1.25,
                kill_height=float(lower[1] - span[1] * 0.5),
            )

    def emit(count: int, fill_volume: bool, camera_index: int) -> None:
        nonlocal positions, velocities, diameters_mm, active, landed
        camera = cam_c2w[min(camera_index, len(cam_c2w) - 1)]
        fx_i, fy_i, cx_i, cy_i = intrinsics[min(camera_index, len(intrinsics) - 1)]
        x = rng.uniform(0, width, count)
        y = rng.uniform(0, height, count) if fill_volume else rng.uniform(-0.08 * height, 0.06 * height, count)
        xi = np.clip(x.astype(np.int32), 0, width - 1)
        yi = np.clip(np.maximum(y, 0).astype(np.int32), 0, height - 1)
        scene_z = depths[min(camera_index, len(depths) - 1)][yi, xi]
        scene_z = np.where(
            np.isfinite(scene_z) & (scene_z > 0) & (scene_z < 999),
            scene_z,
            far_depth,
        )
        z_hi = np.minimum(far_depth, scene_z * 0.86)
        depth_power = 1.0 if config.effect == "rain" else 1.15 if config.effect == "fog" else 1.35
        z = near_depth + (z_hi - near_depth) * np.power(rng.random(count), depth_power)
        camera_points = np.stack(((x - cx_i) / fx_i * z, (y - cy_i) / fy_i * z, z), axis=-1)
        world = camera_points @ camera[:3, :3].T + camera[:3, 3]
        horizontal = rng.uniform(-1, 1, (count, 2)).astype(np.float32)
        particle_diameters = np.clip(
            rng.lognormal(np.log(config.flake_diameter_mm), 0.38, count),
            config.flake_diameter_mm * 0.25,
            config.flake_diameter_mm * 2.2,
        ).astype(np.float32)
        terminal_speed = _terminal_speed(particle_diameters, config.effect)
        speed = (
            -up * terminal_speed[:, None]
            + right * (
                config.wind_metres_per_second
                + horizontal[:, :1] * config.turbulence_metres_per_second * 0.35
            )
            + forward * horizontal[:, 1:] * config.turbulence_metres_per_second * 0.35
        )
        positions = np.concatenate((positions, world.astype(np.float32)))
        velocities = np.concatenate((velocities, speed.astype(np.float32)))
        diameters_mm = np.concatenate((diameters_mm, particle_diameters))
        active = np.concatenate((active, np.ones(count, dtype=bool)))
        landed = np.concatenate((landed, np.zeros(count, dtype=bool)))

    collisions = 0

    def advance(step: int, camera_index: int) -> None:
        nonlocal collisions, positions, velocities, active, landed
        if warp_engine is not None:
            collisions += warp_engine.step(
                positions,
                velocities,
                diameters_mm,
                active,
                landed,
                dt=dt,
                time=step * dt,
                collide=config.collision,
            )
            return
        moving = np.flatnonzero(active)
        if not len(moving):
            return
        local_position = (positions[moving] - origin) @ basis
        phase = (
            step * dt * 0.8
            + local_position[:, 0] * 0.31
            + local_position[:, 1] * 0.17
            + local_position[:, 2] * 0.23
        )
        gust = (
            right * np.sin(phase)[:, None]
            + forward
            * np.cos(phase * 0.73 + local_position[:, 0] * 0.11)[:, None]
        ) * config.turbulence_metres_per_second
        terminal_speed = _terminal_speed(diameters_mm[moving], config.effect)
        target_velocity = (
            -up * terminal_speed[:, None]
            + right * config.wind_metres_per_second
            + gust
        )
        response_seconds = _response_seconds(diameters_mm[moving], config.effect)
        blend = np.minimum(dt / response_seconds, 1.0)
        velocities[moving] += (target_velocity - velocities[moving]) * blend[:, None]
        positions[moving] += velocities[moving] * dt
        query_points, query_normals = surface_points, surface_normals
        query_tree = tree
        if object_collision is not None:
            frame = int(np.clip(camera_index, 0, len(object_collision) - 1))
            extra_points, extra_normals = object_collision[frame]
            if len(extra_points):
                query_points = np.concatenate((surface_points, extra_points))
                query_normals = np.concatenate((surface_normals, extra_normals))
                query_tree = cKDTree(query_points)
        if config.collision:
            distance, nearest = query_tree.query(
                positions[moving], distance_upper_bound=collision_radius, workers=-1
            )
            slots = np.flatnonzero(np.isfinite(distance) & (nearest < len(query_points)))
            if len(slots):
                particles, surfaces = moving[slots], nearest[slots]
                normals = query_normals[surfaces]
                accepted = (
                    ((velocities[particles] @ up) < 0)
                    & ((normals @ up) > 0.12)
                    & (np.einsum("ij,ij->i", velocities[particles], normals) < 0)
                )
                if config.effect in MESH_EFFECTS:
                    accepted &= (normals @ up) < 0.90
                particles, surfaces = particles[accepted], surfaces[accepted]
                positions[particles] = query_points[surfaces] + query_normals[surfaces] * min(
                    collision_radius * 0.1, 0.05
                )
                velocities[particles] = 0
                active[particles] = False
                landed[particles] = True
                collisions += len(particles)
        local = (positions[moving] - origin) @ basis
        active[moving[local[:, 1] < lower[1] - span[1] * 0.5]] = False

    if config.effect in MESH_EFFECTS and config.accumulation and len(mesh_faces):
        settled = seed_settled_on_mesh(
            mesh_verts, mesh_faces, up, rng, config.accumulation_minutes
        )
        if len(settled):
            positions = settled
            velocities = np.zeros_like(settled)
            diameters_mm = np.full(len(settled), config.flake_diameter_mm, dtype=np.float32)
            active = np.zeros(len(settled), dtype=bool)
            landed = np.ones(len(settled), dtype=bool)
    emit(target_particles, True, 0)
    for step in range(warmup_frames):
        emit(emission_rate, False, 0)
        advance(step, 0)
    output = np.empty_like(video)
    for index in range(len(video)):
        emit(emission_rate, False, index)
        advance(warmup_frames + index, index)
        output[index] = _render(
            video[index],
            depths[index],
            cam_c2w[index],
            intrinsics[index],
            positions,
            velocities,
            diameters_mm,
            active,
            landed,
            up,
            surface_points,
            tree,
            dt=dt,
            config=config,
            radius_metres=radius_metres,
            accumulation_strength=accumulation_strength,
            sky_shift=_sky_shift_pixels(
                cam_c2w[index], heading0, up, float(np.mean(intrinsics[index, :2]))
            ),
            sky_matte=sky_mattes[index],
        )
    return SnowResult(
        output,
        np.zeros(video.shape[:3], dtype=bool),
        positions,
        landed,
        len(positions),
        collisions,
        len(surface_points),
        collision_radius,
        frustum_volume,
        target_particles,
        warmup_frames,
        emission_rate,
    )


def simulate_rain(
    video: np.ndarray,
    depths: np.ndarray,
    cam_c2w: np.ndarray,
    intrinsics: np.ndarray,
    static_masks: np.ndarray,
    accumulation_masks: np.ndarray | None = None,
    sky_masks: np.ndarray | None = None,
    *,
    fps: float,
    config: SnowConfig | None = None,
    object_collision: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> SnowResult:
    """Simulate world-space rain with wet surfaces and depth-tested streaks."""
    if config is None:
        config = SnowConfig(
            effect="rain",
            particles_per_cubic_metre=0.07,
            intensity=0.95,
            wind_metres_per_second=1.2,
            turbulence_metres_per_second=0.28,
            flake_diameter_mm=1.6,
            accumulation_minutes=10.0,
            warmup_seconds=1.5,
        )
    elif config.effect != "rain":
        config = replace(config, effect="rain")
    return simulate_snow(
        video,
        depths,
        cam_c2w,
        intrinsics,
        static_masks,
        accumulation_masks,
        sky_masks,
        fps=fps,
        config=config,
        object_collision=object_collision,
    )


def simulate_fog(
    video: np.ndarray,
    depths: np.ndarray,
    cam_c2w: np.ndarray,
    intrinsics: np.ndarray,
    static_masks: np.ndarray,
    accumulation_masks: np.ndarray | None = None,
    sky_masks: np.ndarray | None = None,
    *,
    fps: float,
    config: SnowConfig | None = None,
    object_collision: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> SnowResult:
    """Metric-depth haze: Beer-Lambert in metres, plus sparse slow motes."""
    if config is None:
        config = SnowConfig(
            effect="fog",
            particles_per_cubic_metre=0.012,
            intensity=0.85,
            wind_metres_per_second=0.15,
            turbulence_metres_per_second=0.08,
            flake_diameter_mm=1.2,
            collision=False,
            accumulation=False,
            warmup_seconds=0.5,
            fog_visibility_metres=32.0,
        )
    elif config.effect != "fog":
        config = replace(config, effect="fog")
    return simulate_snow(
        video,
        depths,
        cam_c2w,
        intrinsics,
        static_masks,
        accumulation_masks,
        sky_masks,
        fps=fps,
        config=config,
        object_collision=object_collision,
    )


def simulate_sandstorm(
    video: np.ndarray,
    depths: np.ndarray,
    cam_c2w: np.ndarray,
    intrinsics: np.ndarray,
    static_masks: np.ndarray,
    accumulation_masks: np.ndarray | None = None,
    sky_masks: np.ndarray | None = None,
    *,
    fps: float,
    config: SnowConfig | None = None,
    object_collision: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> SnowResult:
    """Snow flake engine with sand colour, more wind, faster grit."""
    if config is None:
        config = SnowConfig(
            effect="sandstorm",
            particles_per_cubic_metre=0.045,
            intensity=0.85,
            wind_metres_per_second=2.8,
            turbulence_metres_per_second=0.55,
            flake_diameter_mm=1.8,
            accumulation=False,
            warmup_seconds=1.0,
            fog_visibility_metres=70.0,
        )
    elif config.effect != "sandstorm":
        config = replace(config, effect="sandstorm")
    return simulate_snow(
        video,
        depths,
        cam_c2w,
        intrinsics,
        static_masks,
        accumulation_masks,
        sky_masks,
        fps=fps,
        config=config,
        object_collision=object_collision,
    )
