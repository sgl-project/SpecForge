# Export weight-name maps (trainer → SGLang spec-decoder loader)

`specforge export --to sglang` currently handles EAGLE3 checkpoints and renames
trainer-side draft weights to what SGLang's spec-decoder loader expects. Other
strategies use `specforge export --to hf`. Weight-name compatibility is **silent
when wrong** — a key the loader does not recognize is skipped and the draft
serves with uninitialized weights — so the per-architecture map is a documented
artifact, mirrored in code as `specforge/export/to_sglang.py::WEIGHT_MAPS`, and
the exporter validates the produced state against the required-key list rather
than hoping.

## LlamaForCausalLMEagle3 (identity map)

SpecForge's trainer module names are exactly the names sglang's EAGLE3 draft
loader (`sglang/srt/models/llama_eagle3.py`, pinned by `pyproject.toml`) reads, so
`WEIGHT_MAPS["LlamaForCausalLMEagle3"] = {}`:

| Trainer key | SGLang spec-decoder loader key |
|---|---|
| `midlayer.input_layernorm.weight` | same |
| `midlayer.hidden_norm.weight` | same |
| `midlayer.self_attn.{q,k,v,o}_proj.weight` | same |
| `midlayer.post_attention_layernorm.weight` | same |
| `midlayer.mlp.{gate,up,down}_proj.weight` | same |
| `fc.weight` | same |
| `norm.weight` | same |
| `lm_head.weight` | same |
| `t2d`, `d2t` (vocab-mapping buffers) | same |
| `embed_tokens.weight` | **not exported** — serving loads the embedding from the target |

Required keys the exporter enforces: `fc.weight`, `norm.weight`,
`lm_head.weight`, `t2d`, `d2t`.

## MLA (DeepseekV3-style) EAGLE3 draft — to fill when the MLA draft lands

The MLA draft architecture is deferred to its own PR (no reference
implementation is available yet). When it lands, fill the right column by
reading sglang's then-current MLA spec-decoder loader and register the map in
`WEIGHT_MAPS` — do not leave it implicit in code:

| Trainer key | SGLang spec-decoder loader key |
|---|---|
| `midlayer.self_attn.q_a_proj.weight` | TBD |
| `midlayer.self_attn.q_a_layernorm.weight` | TBD |
| `midlayer.self_attn.q_b_proj.weight` | TBD |
| `midlayer.self_attn.kv_a_proj_with_mqa.weight` | TBD |
| `midlayer.self_attn.kv_a_layernorm.weight` | TBD |
| `midlayer.self_attn.kv_b_proj.weight` | TBD |
| `midlayer.self_attn.o_proj.weight` | TBD |
| `fc.weight` / `norm.weight` / `lm_head.weight` / `t2d` / `d2t` | TBD |

## Validation depth

The exporter + tests cover structural correctness (key names, required keys,
tensor round-trip). The full serving round-trip — load the exported draft in a
running sglang speculative-decode server and measure accept length — is a
separate GPU validation step, not covered by the unit gate.
