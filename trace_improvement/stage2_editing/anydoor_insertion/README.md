# AnyDoor insertion

AnyDoor is a published, pretrained diffusion model for inserting a reference object into a
target scene at a chosen location, without training on the specific object or scene. Used
here in place of a flat pixel paste for inserting a person into a drone video.

Produces a more recognizable object shape than a plain pixel paste when viewed closely, but
the result is still small and easy to miss at normal video scale, and needs a Poisson blend
pass afterward to remove a border artifact from the raw model output.

## Run

```bash
modal run pipeline.py::main --step download
modal run pipeline.py::main --step check
modal run pipeline.py::main --step test
modal run pipeline.py::main --step video
```

`test` runs one frame for a quick check. `video` runs all frames.

Needs a reference object image, a mask for that object, and a target frame. See the
`process_pairs` call in `pipeline.py` for the exact input format.
