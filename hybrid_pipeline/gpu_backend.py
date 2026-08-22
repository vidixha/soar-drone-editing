"""Abstract interface for the GPU-backed operations router.py needs.

router.py never imports a specific GPU provider directly -- it only calls
methods on a GPUBackend instance. Any provider (Modal, a local GPU via
subprocess, a different cloud API, a bare-metal box behind a REST endpoint)
can be plugged in by implementing these methods; nothing else in the
router needs to change.

modal_backend.py ships one concrete implementation.
"""


class GPUBackend:
    """Duck-typed interface -- subclass and implement the methods you use.
    All inputs/outputs are raw bytes so a backend can be a remote call, a
    local subprocess, or anything else that can move bytes."""

    def render_trajectory(self, video_bytes: bytes, theta: float, phi: float, r: float,
                          x: float, y: float, video_length: int = 49,
                          sample_h: int = None, sample_w: int = None) -> bytes:
        """Run a depth-reprojection + diffusion gap-fill trajectory edit
        (e.g. TrajectoryCrafter) on video_bytes at the given target camera
        pose (theta=tilt, phi=pan, r=dolly, x/y=lateral). Returns the
        rendered clip as mp4 bytes.

        sample_h/sample_w: optional explicit output resolution. Leave unset
        unless the backend's own default is known to need overriding --
        TrajectoryCrafter's --mode gradual specifically crashes on an
        explicit --sample_size (confirmed at two different resolutions), so
        the reference ModalBackend leaves this unset by default."""
        raise NotImplementedError

    def depth_sequence(self, video_bytes: bytes) -> bytes:
        """Per-frame, temporally-consistent depth for a whole clip in one
        pass (e.g. Trace Anything). Returns a .npz with key 'depth', shape
        (T,H,W), float32, normalized [0,1], higher=closer."""
        raise NotImplementedError

    def depth_single_frame(self, video_bytes: bytes) -> bytes:
        """Single-frame depth snapshot (e.g. Depth Anything V2), used as the
        default (cheaper, less accurate) post-trajectory depth refresh.
        Returns a 16-bit grayscale PNG, higher=closer."""
        raise NotImplementedError

    def remove_objects_inpaint(self, video_bytes: bytes, target: str = None, window: int = 60,
                               stride: int = 3, confidence_thresh: float = 0.6) -> bytes:
        """GPU removal with a generative video-inpainting fallback for the
        case the local/CPU classical path can't handle: an object that never
        exposes clean background in any sampled frame (e.g. a precision-
        tracked shot). Localizes the object by appearance (a generic
        detector, not motion), so it works even where motion-diffing finds
        nothing -- see removal_inpaint_gpu.py's module docstring for the
        full story, including a real bug this fixed (residual mask pointed
        at the wrong region entirely before this). Returns mp4 bytes."""
        raise NotImplementedError

    def apply_weather(self, video_bytes: bytes, kind: str = "snow",
                      intensity: str = "medium", fps: float = 30.0) -> bytes:
        """Reconstruct metric cameras+depth from video_bytes, simulate
        weather (snow/rain/fog/sandstorm), return the weathered clip as
        mp4 bytes. Same reconstruct+simulate path as local
        reconstruction_weather, just remote."""
        raise NotImplementedError


def get_backend(name: str = "modal") -> GPUBackend:
    """Backend factory. Only 'modal' ships today; add another by
    implementing GPUBackend and registering it here."""
    if name == "modal":
        from modal_backend import ModalBackend
        return ModalBackend()
    raise ValueError(f"unknown backend: {name!r} (only 'modal' is implemented)")
