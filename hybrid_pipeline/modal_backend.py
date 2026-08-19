"""Modal implementation of gpu_backend.GPUBackend.

This is one swappable backend, not a dependency of the router itself --
router.py talks to GPUBackend only. Requires these Modal apps to already be
deployed (see each file's own module docstring for exact deploy/download
commands):

  trajcrafter-test      (render_traj)              -- trajcrafter_pipeline.py
  depth-extract         (depth_of_bytes)             -- depth_extract.py
  trace-anything-depth  (depth_sequence_of_bytes)    -- trace_anything_depth.py
  removal-inpaint-gpu   (remove_with_inpaint_fallback) -- removal_inpaint_gpu.py
"""
import modal
from gpu_backend import GPUBackend


class ModalBackend(GPUBackend):
    def render_trajectory(self, video_bytes, theta, phi, r, x, y, video_length=49,
                          sample_h=None, sample_w=None):
        fn = modal.Function.from_name("trajcrafter-test", "render_traj")
        kwargs = {"video_length": video_length}
        if sample_h is not None and sample_w is not None:
            kwargs["sample_h"] = sample_h
            kwargs["sample_w"] = sample_w
        return fn.remote(video_bytes, theta, phi, r, x, y, **kwargs)

    def depth_sequence(self, video_bytes):
        fn = modal.Function.from_name("trace-anything-depth", "depth_sequence_of_bytes")
        return fn.remote(video_bytes)

    def depth_single_frame(self, video_bytes):
        fn = modal.Function.from_name("depth-extract", "depth_of_bytes")
        return fn.remote(video_bytes)

    def remove_objects_inpaint(self, video_bytes, target=None, window=60, stride=3,
                               confidence_thresh=0.6):
        fn = modal.Function.from_name("removal-inpaint-gpu", "remove_with_inpaint_fallback")
        return fn.remote(video_bytes, target=target, window=window, stride=stride,
                         confidence_thresh=confidence_thresh)
