#!/usr/bin/env python3
"""Transform recorded DSV4-Flash traffic into a SpecForge preformatted corpus.

Input:  the traffic-recording JSONL built by the specforge skill
        (rows carry the parsed OpenAI ``request`` plus the SSE-reassembled
        ``response``).
Output: one JSONL row per request: ``{"id", "text", "conv_key", "thinking",
        "finish_reason"}`` where ``text`` is the FULL rendered sequence --
        prompt (system + tools + history) AND the recorded completion --
        encoded with SGLang's native ``encoding_dsv4.encode_messages``, i.e.
        byte-faithful to what the serving path fed and received from the
        target model (DSML tool calls, <tool_result> merging, <think> blocks,
        reasoning-effort preamble).

Train with ``data.is_preformatted: true`` and ``data.chat_template:
deepseek-v4``: SpecForge's parser anchors loss at ``<｜Assistant｜>`` spans,
so all prompt tokens (system/user/tool_result) are loss-masked and only
assistant output (reasoning + content + tool calls + EOS) is supervised.

Why not the ShareGPT ``conversations`` path: the registered deepseek-v4
Jinja renders only system/user/assistant -- ``tool`` messages and
``tool_calls`` vanish silently, and this corpus is ~80% tool-calling.

Row filters (counted, not silent):
  - response_unparsed / missing response  -> skipped
  - finish_reason not in {stop, tool_calls} (length truncations and client
    disconnects have no valid EOS-terminated completion) -> skipped
  - malformed tool_call arguments (mirrors serving's strict normalization)
    -> skipped

thinking_mode is derived from the observed response (usage.reasoning_tokens
>= 1 means the prompt was rendered in thinking mode; the model emits at
least ``</think>``), which captures every upstream toggle -- including
reasoning_effort="none" and chat_template_kwargs -- without trusting the
request (traps 3/4 of the specforge skill).
"""
import argparse
import collections
import glob
import gzip
import json
import os
import sys
from multiprocessing import Pool

# Path to an sglang source tree providing the reference DeepSeek-V4 encoder.
SGLANG_PY = os.environ.get("SF_SGLANG")
if SGLANG_PY:
    sys.path.insert(0, SGLANG_PY)

from sglang.srt.entrypoints.openai import encoding_dsv4  # noqa: E402

REASONING_EFFORT_PROFILE = "official"  # dsv4-flash levels: low/high/max
ACCEPTED_EFFORTS = set(
    encoding_dsv4.REASONING_EFFORT_PROFILES[REASONING_EFFORT_PROFILE]
)
KEEP_FINISH = {"stop", "tool_calls"}


def _flatten_content(content):
    """Serving flattens OpenAI parts-lists to text for dsv4 (string format)."""
    if content is None:
        return ""
    if isinstance(content, list):
        parts = [
            c["text"]
            for c in content
            if isinstance(c, dict) and c.get("type") in ("text", "input_text")
        ]
        return " ".join(parts) if parts else ""
    return content


def _normalize_tool_calls(tool_calls):
    """Strict normalization mirroring serving: arguments must parse to a dict.

    Returns the normalized list, or raises ValueError on malformed input.
    """
    out = []
    for tc in tool_calls or []:
        fn = (tc or {}).get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            parsed = json.loads(args)  # raises on malformed
        else:
            parsed = args
        if not isinstance(parsed, dict):
            raise ValueError("tool call arguments must be a JSON object")
        out.append(
            {
                "id": tc.get("id") or "",
                "type": tc.get("type") or "function",
                "function": {"name": fn.get("name") or "", "arguments": parsed},
            }
        )
    return out


def build_text(row):
    """-> (out_row or None, skip_reason or None)"""
    if row.get("response_unparsed"):
        return None, "response_unparsed"
    resp = row.get("response") or {}
    choices = resp.get("choices") or []
    if not choices:
        return None, "no_choices"
    choice = choices[0]
    finish = choice.get("finish_reason")
    if finish not in KEEP_FINISH:
        return None, f"finish_{finish}"
    msg = choice.get("message") or {}
    req = row.get("request") or {}
    usage = resp.get("usage") or {}

    reasoning_tokens = usage.get("reasoning_tokens") or 0
    thinking = bool(reasoning_tokens >= 1 or msg.get("reasoning_content"))
    thinking_mode = "thinking" if thinking else "chat"

    messages = []
    for m in req.get("messages") or []:
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool", "developer"):
            return None, f"role_{role}"
        nm = {"role": role, "content": _flatten_content(m.get("content"))}
        if role == "tool":
            nm["tool_call_id"] = m.get("tool_call_id", "")
        if role == "assistant":
            if m.get("reasoning_content"):
                nm["reasoning_content"] = m["reasoning_content"]
            if m.get("tool_calls"):
                nm["tool_calls"] = _normalize_tool_calls(m["tool_calls"])
        messages.append(nm)

    final = {
        "role": "assistant",
        "content": msg.get("content") or "",
        "reasoning_content": msg.get("reasoning_content") or "",
    }
    if msg.get("tool_calls"):
        final["tool_calls"] = _normalize_tool_calls(msg["tool_calls"])
    messages.append(final)

    # Serving inserts an empty system turn to host the tools block.
    if not messages or messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    if req.get("tools"):
        messages[0]["tools"] = req["tools"]
    if req.get("response_format"):
        messages[0]["response_format"] = req["response_format"]

    eff = req.get("reasoning_effort")
    v4_eff = eff if eff in ACCEPTED_EFFORTS else None

    text = encoding_dsv4.encode_messages(
        messages,
        thinking_mode=thinking_mode,
        reasoning_effort=v4_eff,
        reasoning_effort_profile=REASONING_EFFORT_PROFILE,
    )
    return (
        {
            "id": row.get("request_id") or "",
            "text": text,
            "conv_key": row.get("_conv_key") or "",
            "thinking": thinking,
            "finish_reason": finish,
        },
        None,
    )


def process_shard(arg):
    shard_path, out_path = arg
    st = collections.Counter()
    with gzip.open(shard_path, "rt", encoding="utf-8") as fin, open(
        out_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            st["rows_in"] += 1
            try:
                row = json.loads(line)
                out, skip = build_text(row)
            except Exception as e:  # malformed tool args, encoder errors
                st[f"err_{type(e).__name__}"] += 1
                continue
            if out is None:
                st[f"skip_{skip}"] += 1
                continue
            try:
                blob = json.dumps(out, ensure_ascii=False)
            except (UnicodeEncodeError, ValueError):
                blob = json.dumps(out, ensure_ascii=True)
                st["surrogate_escaped"] += 1
            try:
                fout.write(blob + "\n")
            except UnicodeEncodeError:
                fout.write(json.dumps(out, ensure_ascii=True) + "\n")
                st["surrogate_escaped"] += 1
            st["rows_out"] += 1
            st["text_chars"] += len(out["text"])
            st["thinking_rows"] += bool(out["thinking"])
    return dict(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir of *.jsonl.gz traffic shards")
    ap.add_argument("--out", required=True, help="output .jsonl path")
    ap.add_argument("--workers", type=int, default=40)
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.src, "*.jsonl.gz")))
    if not shards:
        raise SystemExit(f"no shards under {args.src}")
    parts_dir = args.out + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    jobs = [
        (s, os.path.join(parts_dir, f"part-{i:04d}.jsonl"))
        for i, s in enumerate(shards)
    ]
    with Pool(min(args.workers, len(jobs))) as p:
        stats = p.map(process_shard, jobs)

    agg = collections.Counter()
    for st in stats:
        agg.update(st)

    # one writer merges the parts -- never fan parallel writers into one file
    with open(args.out, "w", encoding="utf-8") as fout:
        for _, part in jobs:
            with open(part, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
            os.remove(part)
    os.rmdir(parts_dir)

    for k in sorted(agg):
        print(f"  {k} = {agg[k]}")
    kept = agg["rows_out"]
    total = agg["rows_in"]
    print(f"DONE {kept}/{total} rows ({kept / max(total, 1):.1%}) -> {args.out}")


if __name__ == "__main__":
    main()
