"""NVIDIA Warp flake engine: GPU particles vs a reconstructed depth mesh."""

from __future__ import annotations

import numpy as np

_wp = None
_kernels = None


def available() -> bool:
    try:
        import warp as wp

        wp.init()
    except Exception:
        return False
    return True


def _warp():
    global _wp, _kernels
    if _wp is not None:
        return _wp, _kernels
    import warp as wp

    wp.init()

    @wp.kernel
    def step_flakes(
        positions: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec3),
        diameters: wp.array(dtype=float),
        active: wp.array(dtype=wp.int32),
        landed: wp.array(dtype=wp.int32),
        mesh: wp.uint64,
        up: wp.vec3,
        right: wp.vec3,
        forward: wp.vec3,
        wind: float,
        turbulence: float,
        dt: float,
        time: float,
        collide: wp.int32,
        max_dist: float,
        kill_height: float,
        origin: wp.vec3,
        hits: wp.array(dtype=wp.int32),
    ):
        index = wp.tid()
        if active[index] == 0:
            return
        position = positions[index]
        velocity = velocities[index]
        diameter = diameters[index]
        terminal = 0.55 + 0.38 * wp.pow(diameter, 0.55)
        phase = time * 0.8 + position[0] * 0.31 + position[1] * 0.17 + position[2] * 0.23
        gust = (right * wp.sin(phase) + forward * wp.cos(phase * 0.73 + position[0] * 0.11)) * turbulence
        target = -up * terminal + right * wind + gust
        response = wp.clamp(0.18 + diameter * 0.08, 0.2, 0.8)
        velocity = velocity + (target - velocity) * wp.min(dt / response, 1.0)
        predicted = position + velocity * dt
        if collide == 1:
            query = wp.mesh_query_point_no_sign(mesh, predicted, max_dist)
            if query.result:
                hit = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
                normal = wp.mesh_eval_face_normal(mesh, query.face)
                if wp.dot(normal, up) < 0.0:
                    normal = -normal
                facing = wp.dot(normal, up)
                approaching = wp.dot(velocity, normal) < 0.0 and wp.dot(velocity, up) < 0.0
                if approaching and facing > 0.16:
                    predicted = hit + normal * 0.04
                    velocity = wp.vec3(0.0, 0.0, 0.0)
                    active[index] = 0
                    landed[index] = 1
                    wp.atomic_add(hits, 0, 1)
        if wp.dot(predicted - origin, up) < kill_height:
            active[index] = 0
        positions[index] = predicted
        velocities[index] = velocity

    _wp = wp
    _kernels = {"step_flakes": step_flakes}
    return _wp, _kernels


def build_depth_mesh(
    depths: np.ndarray,
    cameras: np.ndarray,
    intrinsics: np.ndarray,
    sky_masks: np.ndarray,
    *,
    stride: int = 6,
    frame_stride: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangle mesh from AerialMetric depth. Sky pixels never become vertices."""
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for index in range(0, len(depths), max(1, frame_stride)):
        keep = ~sky_masks[index]
        verts, tris = _frame_depth_mesh(depths[index], cameras[index], intrinsics[index], keep, stride)
        if len(verts) == 0:
            continue
        vertices.append(verts)
        faces.append(tris + offset)
        offset += len(verts)
    if not vertices:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)
    return np.concatenate(vertices), np.concatenate(faces)


def _frame_depth_mesh(
    depth: np.ndarray,
    camera: np.ndarray,
    intrinsics: np.ndarray,
    keep: np.ndarray,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    step = max(2, stride)
    rows = (height - 1) // step + 1
    cols = (width - 1) // step + 1
    ys = np.clip(np.arange(rows) * step, 0, height - 1)
    xs = np.clip(np.arange(cols) * step, 0, width - 1)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    z = depth[grid_y, grid_x].astype(np.float32)
    valid = keep[grid_y, grid_x] & np.isfinite(z) & (z > 0) & (z < 999)
    fx, fy, cx, cy = intrinsics.astype(np.float32)
    cam = np.stack(
        ((grid_x.astype(np.float32) - cx) / fx * z, (grid_y.astype(np.float32) - cy) / fy * z, z),
        axis=-1,
    )
    world = cam @ camera[:3, :3].astype(np.float32).T + camera[:3, 3].astype(np.float32)
    index = np.full((rows, cols), -1, dtype=np.int32)
    index[valid] = np.arange(int(valid.sum()), dtype=np.int32)
    quad = (
        valid[:-1, :-1]
        & valid[:-1, 1:]
        & valid[1:, :-1]
        & valid[1:, 1:]
    )
    z00, z01, z10, z11 = z[:-1, :-1], z[:-1, 1:], z[1:, :-1], z[1:, 1:]
    zmin = np.minimum(np.minimum(z00, z01), np.minimum(z10, z11))
    zmax = np.maximum(np.maximum(z00, z01), np.maximum(z10, z11))
    quad &= (zmax - zmin) < np.maximum(zmin * 0.08, 1.2)
    i00, i01 = index[:-1, :-1][quad], index[:-1, 1:][quad]
    i10, i11 = index[1:, :-1][quad], index[1:, 1:][quad]
    tris = np.stack((i00, i01, i11, i00, i11, i10), axis=-1).reshape(-1, 3)
    return world[valid].astype(np.float32), tris.astype(np.int32)


def seed_settled_on_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    up: np.ndarray,
    rng: np.random.Generator,
    minutes: float,
    *,
    max_points: int = 14_000,
) -> np.ndarray:
    """World-space settled flakes on upward mesh faces. No image mask."""
    if len(vertices) == 0 or len(faces) == 0 or minutes <= 0:
        return np.empty((0, 3), dtype=np.float32)
    corners = vertices[faces]
    edge_a = corners[:, 1] - corners[:, 0]
    edge_b = corners[:, 2] - corners[:, 0]
    crossed = np.cross(edge_a, edge_b)
    area = 0.5 * np.linalg.norm(crossed, axis=1)
    lengths = np.linalg.norm(crossed, axis=1, keepdims=True)
    normals = np.divide(crossed, np.clip(lengths, 1e-8, None))
    normals[(normals @ up) < 0] *= -1
    facing = normals @ up
    keep = (facing > 0.18) & (area > 1e-4)
    if int(keep.sum()) > 32:
        large = area > np.percentile(area[keep], 70)
        keep &= ~((facing > 0.84) & large)
    if not keep.any():
        return np.empty((0, 3), dtype=np.float32)
    area = area[keep]
    corners = corners[keep]
    density = 0.35 + 1.8 * float(np.clip(minutes / 16.0, 0, 1))
    count = int(np.clip(area.sum() * density, 0, max_points))
    if count == 0:
        return np.empty((0, 3), dtype=np.float32)
    pick = rng.choice(len(area), size=count, replace=True, p=area / area.sum())
    u = rng.random(count)
    v = rng.random(count)
    fold = u + v > 1.0
    u[fold] = 1.0 - u[fold]
    v[fold] = 1.0 - v[fold]
    points = (
        corners[pick, 0]
        + (corners[pick, 1] - corners[pick, 0]) * u[:, None]
        + (corners[pick, 2] - corners[pick, 0]) * v[:, None]
    )
    return points.astype(np.float32)


class WarpFlakeEngine:
    """PBD-style flakes colliding with a Warp triangle mesh. No bounce."""

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        up: np.ndarray,
        right: np.ndarray,
        forward: np.ndarray,
        origin: np.ndarray,
        wind: float,
        turbulence: float,
        max_dist: float,
        kill_height: float,
    ) -> None:
        wp, kernels = _warp()
        self.wp = wp
        self.kernel = kernels["step_flakes"]
        self.device = "cuda:0" if wp.is_cuda_available() else "cpu"
        self.mesh = wp.Mesh(
            points=wp.array(np.ascontiguousarray(vertices), dtype=wp.vec3, device=self.device),
            indices=wp.array(np.ascontiguousarray(faces.reshape(-1)), dtype=wp.int32, device=self.device),
        )
        self.up = wp.vec3(*np.asarray(up, dtype=np.float32).tolist())
        self.right = wp.vec3(*np.asarray(right, dtype=np.float32).tolist())
        self.forward = wp.vec3(*np.asarray(forward, dtype=np.float32).tolist())
        self.origin = wp.vec3(*np.asarray(origin, dtype=np.float32).tolist())
        self.wind = float(wind)
        self.turbulence = float(turbulence)
        self.max_dist = float(max_dist)
        self.kill_height = float(kill_height)

    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        diameters_mm: np.ndarray,
        active: np.ndarray,
        landed: np.ndarray,
        *,
        dt: float,
        time: float,
        collide: bool,
    ) -> int:
        if len(positions) == 0 or not active.any():
            return 0
        wp = self.wp
        pos = wp.array(np.ascontiguousarray(positions), dtype=wp.vec3, device=self.device)
        vel = wp.array(np.ascontiguousarray(velocities), dtype=wp.vec3, device=self.device)
        diam = wp.array(np.ascontiguousarray(diameters_mm), dtype=float, device=self.device)
        live = wp.array(active.astype(np.int32), dtype=wp.int32, device=self.device)
        stuck = wp.array(landed.astype(np.int32), dtype=wp.int32, device=self.device)
        hits = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(
            self.kernel,
            dim=len(positions),
            inputs=[
                pos,
                vel,
                diam,
                live,
                stuck,
                self.mesh.id,
                self.up,
                self.right,
                self.forward,
                self.wind,
                self.turbulence,
                float(dt),
                float(time),
                1 if collide else 0,
                self.max_dist,
                self.kill_height,
                self.origin,
                hits,
            ],
            device=self.device,
        )
        wp.synchronize()
        positions[:] = pos.numpy()
        velocities[:] = vel.numpy()
        active[:] = live.numpy().astype(bool)
        landed[:] = stuck.numpy().astype(bool)
        return int(hits.numpy()[0])
