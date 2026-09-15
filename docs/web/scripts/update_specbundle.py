#!/usr/bin/env python3
"""Refresh ``docs/data/specbundle_models.json`` from the Hugging Face SpecBundle collection.

The JSON file is the source of truth for the SpecBundle gallery and can be
edited by hand. This script only adds models that joined the Hugging Face
collection and refreshes the counters (downloads, likes, parameters, last
modified). Anything curated in the file survives a run:

* ``target``   the model the draft was trained for
* ``dataset``  the regenerated training dataset, if published
* ``provider.fullname`` when the Hugging Face profile has no display name

For a brand-new model the target is looked up from the model card's
``base_model`` metadata; when that is missing the script prints the model so
someone can fill in ``target`` by hand.

Usage::

    python3 docs/web/scripts/update_specbundle.py
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

COLLECTION_API = "https://huggingface.co/api/collections/lmsys/specbundle"
COLLECTION_URL = "https://huggingface.co/collections/lmsys/specbundle"
OUTPUT = Path(__file__).resolve().parents[2] / "data" / "specbundle_models.json"

METHOD_PATTERNS = [
    ("DSpark", re.compile(r"dspark", re.I)),
    ("Domino", re.compile(r"domino", re.I)),
    ("DFlash", re.compile(r"dflash", re.I)),
    ("EAGLE3", re.compile(r"eagle-?3|eagle", re.I)),
]


def detect_method(repo_id: str, tags: list[str]) -> str:
    for name, pattern in METHOD_PATTERNS:
        if pattern.search(repo_id):
            return name
    for name, pattern in METHOD_PATTERNS:
        if any(pattern.search(t) for t in tags):
            return name
    return "Other"


def fetch_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "specforge-docs"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def base_model_from_api(repo_id: str) -> str | None:
    """Look up ``base_model`` from the model card metadata when it is set."""
    try:
        info = fetch_json(f"https://huggingface.co/api/models/{repo_id}")
    except Exception as exc:  # network hiccup: leave the target for a human
        print(f"  warn: could not fetch {repo_id}: {exc}", file=sys.stderr)
        return None
    base = (info.get("cardData") or {}).get("base_model")
    if isinstance(base, list):
        base = base[0] if base else None
    if not base:
        for tag in info.get("tags", []):
            if tag.startswith("base_model:") and ":" not in tag[len("base_model:") :]:
                base = tag[len("base_model:") :]
                break
    return base


def load_existing(path: Path) -> dict[str, dict]:
    """Models already in the file, keyed by repo id."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        sys.exit(f"{path} is not valid JSON: {exc}")
    return {m["id"]: m for m in data.get("models", []) if m.get("id")}


def main() -> None:
    existing = load_existing(OUTPUT)
    print(f"Loaded {len(existing)} models from {OUTPUT}")
    print(f"Fetching {COLLECTION_API}")
    collection = fetch_json(COLLECTION_API)

    models = []
    seen: set[str] = set()
    for item in collection.get("items", []):
        if item.get("type") != "model":
            continue
        repo_id = item["id"]
        seen.add(repo_id)
        prev = existing.get(repo_id, {})
        prev_provider = prev.get("provider") or {}
        author = item.get("authorData") or {}
        provider_key = item.get("author") or repo_id.split("/")[0]

        target = prev.get("target") or base_model_from_api(repo_id)
        models.append(
            {
                "id": repo_id,
                "name": repo_id.split("/", 1)[1],
                "provider": {
                    "name": provider_key,
                    "fullname": prev_provider.get("fullname")
                    or author.get("fullname")
                    or provider_key,
                    "avatar": author.get("avatarUrl") or prev_provider.get("avatar"),
                    "type": author.get("type") or prev_provider.get("type") or "user",
                },
                "method": prev.get("method")
                or detect_method(repo_id, item.get("tags", [])),
                "target": target,
                "dataset": prev.get("dataset"),
                "downloads": item.get("downloads", prev.get("downloads", 0)),
                "likes": item.get("likes", prev.get("likes", 0)),
                "numParameters": item.get("numParameters") or prev.get("numParameters"),
                "lastModified": item.get("lastModified") or prev.get("lastModified"),
                "gated": bool(item.get("gated")),
            }
        )
        flag = "" if repo_id in existing else "  (new)"
        print(f"  {repo_id} -> {target or '?'}{flag}")

    # Keep hand-added models that are not (yet) in the Hugging Face collection.
    for repo_id, prev in existing.items():
        if repo_id not in seen:
            models.append(prev)
            print(
                f"  {repo_id} -> {prev.get('target') or '?'}  (kept, not in collection)"
            )

    unknown = [m["id"] for m in models if not m.get("target")]
    if unknown:
        print(
            "\nModels without a known target model (set `target` in the JSON):\n  "
            + "\n  ".join(unknown),
            file=sys.stderr,
        )

    payload = {
        "collection": COLLECTION_URL,
        "collectionApi": COLLECTION_API,
        "title": collection.get("title", "SpecBundle"),
        "description": collection.get("description", ""),
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": models,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(models)} models to {OUTPUT}")


if __name__ == "__main__":
    main()
