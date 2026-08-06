# Classical compositing

Moves or inserts an object in a video without training a model, using inpainting and
pixel blending only.

## Files

- `erase_and_reinsert_v1.py` first attempt, uses OpenCV's built in inpainting to erase an
  object and a flat pixel paste to put it back elsewhere. Works, but leaves a visible mark
  where the object was and a soft edge around the pasted object.
- `lama_erase.py` replaces OpenCV's inpainting with LaMa, a pretrained inpainting model,
  run on Modal. Removes the visible mark from v1.
- `erase_and_reinsert_v2_poisson.py` replaces the flat paste with Poisson blending
  (`cv2.seamlessClone`), removing the soft edge from v1. Use this together with
  `lama_erase.py` for the best result.
- `insert_person.py` inserts a real person, cropped from real footage, walking along a
  chosen path. Works as a proof of the compositing mechanics, but the person is too small
  to read clearly as a person once resized down to scene scale.

## Run

```bash
modal run lama_erase.py::main
python erase_and_reinsert_v2_poisson.py
python insert_person.py
```
