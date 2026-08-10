# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Online-live ref source: capture records pushed by external serving engines.

The driven transport (:class:`SGLangServerCaptureAdapter`) originates capture
requests and reads results from ``/generate`` responses.  In live mode the
serving engine originates samples from real user traffic, writes tensors into
Mooncake itself, and pushes the same result record to the producer's
:class:`CaptureIntakeServer`.  This source is the receiving half: it publishes
the capture-config handshake document and turns each validated record into a
committed-ready ``SampleRef``.
"""

from __future__ import annotations

from typing import Any, Dict, List

from specforge.inference.adapters.server_capture import (
    ServerCaptureSchema,
    build_server_capture_ref,
    validate_server_capture_ref,
)
from specforge.inference.capture import CaptureConfig, CaptureMismatchError

_LIVE_TRANSPORT = "sglang_live_capture"


class LiveIntakeRefSource:
    """Validate pushed capture records against the run's capture contract."""

    def __init__(
        self,
        store,
        *,
        run_id: str,
        algorithm: str,
        schema: ServerCaptureSchema,
        capture: CaptureConfig,
        max_num_tokens: int,
        target_model_version: str = "unknown",
    ) -> None:
        if not hasattr(store, "adopt") or not hasattr(store, "store_id"):
            raise TypeError(
                "LiveIntakeRefSource needs a MooncakeFeatureStore-like store"
            )
        if not algorithm:
            raise ValueError("algorithm must be non-empty")
        if max_num_tokens < 1:
            raise ValueError("max_num_tokens must be >= 1")
        self.store = store
        self.run_id = run_id
        self.strategy = algorithm
        self.schema = schema
        self.capture = capture
        self.max_num_tokens = int(max_num_tokens)
        self.target_model_version = target_model_version
        self._expected_features = frozenset(self._feature_names())

    def _feature_names(self) -> List[str]:
        names = [
            name
            for name in (self.schema.aux_feature, self.schema.last_hidden_feature)
            if name is not None
        ]
        names.extend(name for name, _key, _trailing in self.schema.passthrough)
        if self.schema.attention_mask_feature is not None:
            names.append(self.schema.attention_mask_feature)
        return names

    def config_payload(self) -> Dict[str, Any]:
        """The handshake document a live capture server GETs at startup."""
        features: Dict[str, str] = {}
        if self.schema.aux_feature is not None:
            features["aux"] = self.schema.aux_feature
        if self.schema.last_hidden_feature is not None:
            features["last_hidden"] = self.schema.last_hidden_feature
        passthrough: List[Dict[str, str]] = []
        for feature_name, payload_key, trailing in self.schema.passthrough:
            if trailing:
                raise ValueError(
                    f"live capture cannot synthesize passthrough {feature_name!r} "
                    f"with trailing shape {trailing}"
                )
            source = "tokens" if payload_key == "input_ids" else "ones"
            passthrough.append({"name": feature_name, "source": source})
        if self.schema.attention_mask_feature is not None:
            passthrough.append(
                {"name": self.schema.attention_mask_feature, "source": "ones"}
            )
        return {
            "store_id": str(self.store.store_id),
            "run_id": self.run_id,
            "gen": 1,
            "features": features,
            "passthrough": passthrough,
            # Floor of 2 keeps warmup/probe one-token requests out of training
            # (DFlash-family objectives need two consecutive supervised tokens).
            "min_num_tokens": 2,
            "max_num_tokens": self.max_num_tokens,
        }

    def ref_from_record(self, record: Dict[str, Any]):
        """Build and verify a SampleRef; raises loudly on any contract breach."""
        store_id = str(record.get("store_id"))
        if store_id != str(self.store.store_id):
            raise ValueError(
                f"record store_id {store_id!r} != run store "
                f"{self.store.store_id!r}"
            )
        if int(record.get("gen", -1)) != 1:
            raise ValueError(f"live capture requires gen=1, got {record.get('gen')}")
        num_tokens = int(record.get("num_tokens", 0))
        if num_tokens < 2:
            raise ValueError("record requires num_tokens >= 2")
        if num_tokens > self.max_num_tokens:
            raise ValueError(
                f"record num_tokens {num_tokens} exceeds the run cap "
                f"{self.max_num_tokens}"
            )
        features = record.get("features")
        if not isinstance(features, dict) or not features:
            raise ValueError("record requires a features mapping")
        if frozenset(features) != self._expected_features:
            raise CaptureMismatchError(
                f"record features {sorted(features)} != expected "
                f"{sorted(self._expected_features)}"
            )
        ref = build_server_capture_ref(
            record,
            schema=self.schema,
            capture=self.capture,
            run_id=self.run_id,
            strategy=self.strategy,
            source_task_id=str(record["sample_id"]),
            target_model_version=self.target_model_version,
            num_tokens=num_tokens,
            transport=_LIVE_TRANSPORT,
            origin=str(record.get("origin", "live")),
        )
        validate_server_capture_ref(
            ref,
            record,
            schema=self.schema,
            capture=self.capture,
            expected_len=num_tokens,
        )
        return ref


__all__ = ["LiveIntakeRefSource"]
