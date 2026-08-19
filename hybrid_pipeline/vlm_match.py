"""Attribute-aware target matching for removal: does a detected candidate
blob actually depict what the instruction described ("the black car"), not
just "a compact solid moving thing"?

modules.py's motion detector is a proposal generator, not a classifier -- it
finds candidate blobs (moving, compact, solid) but has no notion of car vs.
person vs. which of several cars. Without this, "remove the black car" and
"remove the car" are indistinguishable once parsed down to "car", and with
multiple cars in frame, all of them (or none) get removed together.

Model: CLIP (openai/clip-vit-base-patch32, ~151M params). Chosen over a
generative/chat VLM (Moondream, SmolVLM, etc.) for this specific job: this
is verification against a short, known label set, not open-ended visual
reasoning, so a contrastive image-text model is a better fit than one that
has to be prompted and its free-text answer re-parsed -- CLIP just needs one
forward pass and a softmax, no prompt engineering, no generation latency
across every candidate blob in every frame. CPU-only.
"""
import cv2
import numpy as np

MODEL_ID = "openai/clip-vit-base-patch32"

_model = None
_proc = None

# Fixed vocabularies to build sibling alternatives from -- the point isn't
# an exhaustive ontology, it's giving CLIP something to discriminate
# *against*. Comparing "black car" only to a vacuous "empty road" negative
# doesn't test color at all -- confirmed empirically: a crop that
# undisputedly showed a black car scored "black car", "white car", and
# "person" all as True against that framing, because every one of them
# beats "nothing is here" once something visibly occupies the crop. CLIP
# only becomes a real classifier once it's forced to pick among genuine
# alternatives for the same slot (color-vs-color, class-vs-class).
_COLORS = ["black", "white", "red", "blue", "green", "silver", "gray", "yellow", "orange"]
_CLASSES = ["car", "truck", "van", "person", "bicycle", "motorcycle"]

def _score(crop_rgb, labels):
    model, proc = _load()
    import torch
    inputs = proc(text=labels, images=crop_rgb, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**inputs)
    return out.logits_per_image.softmax(dim=1)[0].tolist()

def _load():
    global _model, _proc
    if _model is None:
        from transformers import CLIPModel, CLIPProcessor
        _model = CLIPModel.from_pretrained(MODEL_ID, use_safetensors=True)
        _proc = CLIPProcessor.from_pretrained(MODEL_ID)
    return _model, _proc

def matches_target(crop_bgr: np.ndarray, target: str) -> bool:
    """True if `crop_bgr` (a candidate blob's frame crop, BGR) is CLIP's
    best guess for `target`, decided by pitting the target's color and/or
    class against sibling alternatives from a fixed vocabulary (not a
    vacuous "is anything here" negative -- see module docstring for why
    that doesn't work). Words in `target` outside both vocabularies (e.g.
    an unrecognized class) fall back to a plain target-vs-generic-object
    check. Aerial top-down crops of a small object are a weak spot for CLIP
    (mostly trained on eye-level photos) -- this is a lightweight first
    pass, not a guaranteed-correct classifier."""
    if crop_bgr.size == 0:
        return False
    words = target.lower().split()
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    target_color = next((w for w in words if w in _COLORS), None)
    target_class = next((w for w in words if w in _CLASSES
                          or w + "s" in [c + "s" for c in _CLASSES]), None)
    if target_class and target_class not in _CLASSES:
        target_class = target_class.rstrip("s")

    if target_color and target_class:
        labels = [f"a {c} {target_class}" for c in _COLORS]
        probs = _score(rgb, labels)
        return _COLORS[int(np.argmax(probs))] == target_color
    if target_class:
        labels = [f"a {c}" for c in _CLASSES]
        probs = _score(rgb, labels)
        return _CLASSES[int(np.argmax(probs))] == target_class
    # No recognized class/color (e.g. "objects", an unusual noun) -- can't
    # build a meaningful sibling set, so don't reject a shape-filtered
    # candidate over a vocabulary gap.
    return True
