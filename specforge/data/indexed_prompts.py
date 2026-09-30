"""Bounded-memory access to an immutable, fully admitted token JSONL corpus."""

import hashlib
import json
import os
import sys
from array import array
from collections.abc import Sequence
from pathlib import Path


class IndexedPromptSequence(Sequence):
    """Preserve admitted row order while normalizing only requested rows.

    Admission must have scanned every row with the recorded context/filter
    contract. This index does not replace that scan or silently filter rows.
    Both files are hashed before use, and every retrieved row is validated.
    """

    def __init__(
        self,
        path,
        manifest_path,
        *,
        max_length,
        min_loss_tokens,
        max_prompts=None,
        loss_mask_filter=None,
    ):
        from .prompt_builder import _prompt_from_record

        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        expected_filter = loss_mask_filter.__name__ if loss_mask_filter else None
        if (
            manifest["schema_version"] != 1
            or manifest["max_length"] != max_length
            or manifest["min_loss_tokens"] != min_loss_tokens
            or manifest["loss_mask_filter"] != expected_filter
        ):
            raise ValueError(
                "indexed corpus admission contract does not match training"
            )
        offsets_data = (manifest_path.parent / manifest["offsets_file"]).read_bytes()
        if hashlib.sha256(offsets_data).hexdigest() != manifest["offsets_sha256"]:
            raise ValueError("prompt offset digest mismatch")
        offsets = array("Q")
        offsets.frombytes(offsets_data)
        if sys.byteorder != "little":
            offsets.byteswap()
        if (
            len(offsets) != manifest["records"] + 1
            or not offsets
            or offsets[0] != 0
            or offsets[-1] != manifest["source_bytes"]
            or any(a >= b for a, b in zip(offsets, offsets[1:]))
        ):
            raise ValueError("invalid prompt offsets")
        self._path = os.fspath(path)
        with open(path, "rb") as stream:
            source_hash = hashlib.sha256()
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                source_hash.update(chunk)
            stat = os.fstat(stream.fileno())
        if (
            stat.st_size != manifest["source_bytes"]
            or source_hash.hexdigest() != manifest["source_sha256"]
        ):
            raise ValueError("admitted prompt corpus digest mismatch")
        self._identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        self._offsets = offsets
        self._length = manifest["records"]
        if max_prompts:
            self._length = min(self._length, max_prompts)
        self._max_length = max_length
        self._min_loss_tokens = min_loss_tokens
        self._loss_mask_filter = loss_mask_filter
        self._normalize = _prompt_from_record

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        with open(self._path, "rb") as stream:
            stat = os.fstat(stream.fileno())
            if (
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
            ) != self._identity:
                raise ValueError("admitted prompt file changed after indexing")
            stream.seek(self._offsets[index])
            raw = stream.read(self._offsets[index + 1] - self._offsets[index])
        prompt = self._normalize(
            json.loads(raw),
            source=f"{self._path}:row {index}",
            max_length=self._max_length,
            min_loss_tokens=self._min_loss_tokens,
        )
        if prompt is None or (
            self._loss_mask_filter
            and not self._loss_mask_filter(prompt["payload"]["loss_mask"])
        ):
            raise ValueError(f"indexed prompt row {index} violates frozen admission")
        return prompt
