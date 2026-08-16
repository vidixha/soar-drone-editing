"""Small-LLM instruction parser: NL -> the same op-list schema parse() (the
regex parser) produces, but generalizes past an exact keyword list.

Model: Qwen2.5-0.5B-Instruct (494M params). Chosen deliberately small --
this is a structured-extraction task (verb + target + a couple of params
from one short imperative sentence), not open-ended reasoning, so a
frontier LLM would be real cost/latency for no accuracy benefit. Runs
CPU-only in bfloat16 (~1.6GB peak RSS measured), no GPU needed.

Validated against the regex parser on a paraphrase with zero keyword
overlap ("get the vehicles out of the frame please") -- the LLM correctly
produced remove(vehicles); the regex parser cannot handle this by
construction, since it only matches its fixed keyword list.

Tradeoff to know: ~15-25s per call on 2 CPU cores (model load is cached
after the first call in a process). Negligible next to a multi-minute GPU
trajectory render, but real, noticeable overhead for a removal-only
instruction that would otherwise be instant -- this is exactly why it's
opt-in (--parser llm), not the default.
"""
import json, re

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Kept deliberately minimal -- confirmed empirically that adding more
# clauses to this prompt ("no markdown fences", scheduling guidance, an
# empty-array fallback instruction) measurably degrades this 494M model's
# ability to follow the core schema: the exact same instruction that parses
# correctly here produced a bare ["vehicle"] (wrong shape) with a longer,
# more "thorough" version of this prompt. A model this small has very
# little capacity to track multiple simultaneous instructions -- every
# extra clause is a real tax on it, unlike with a frontier model.
SYSTEM_PROMPT = """You convert a natural-language video-editing instruction into a JSON list of operations.
Valid modules and their params:
- {"module":"remove","params":{"target":"<object noun>"}}
- {"module":"trajectory","params":{"motion":"orbit|pan|dolly|fly|zoom|crane","dir":"left|right|forward|backward|in|out|up|down"}}
- {"module":"insert","params":{"object":"car|vehicle|truck|person|object"}}
Output ONLY a JSON array, no prose. Order operations as they should logically run."""

_model = None
_tok = None

def _load():
    global _model, _tok
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    return _model, _tok

def _extract_json(text):
    """Model output is usually a clean JSON array, sometimes wrapped in a
    ```json fence, occasionally with a stray sentence around it -- pull out
    the first [...] block rather than assuming exact formatting."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence: return fence.group(1)
    bracket = re.search(r"\[.*\]", text, re.DOTALL)
    if bracket: return bracket.group(0)
    return text

def parse_llm(text: str) -> list:
    """Same return shape as router.parse(): a list of {"module","params"}
    dicts. Raises ValueError on unparseable model output rather than
    silently returning something wrong -- a bad parse should be visible,
    not quietly executed."""
    model, tok = _load()
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                         temperature=None, top_p=None, pad_token_id=tok.eos_token_id)
    raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        ops = json.loads(_extract_json(raw))
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM parser produced invalid JSON: {raw!r}") from e
    if not isinstance(ops, list) or not all(isinstance(o, dict) and "module" in o and "params" in o for o in ops):
        raise ValueError(f"LLM parser produced malformed op-list: {ops!r}")
    return ops
