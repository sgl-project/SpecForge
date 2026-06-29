# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Two-axis staleness for online (disaggregated) training.

In an online loop the trainer keeps producing fresh draft weights while the
rollout pool keeps generating samples; without hot-swapping the rollout (out of
scope here — no hot-update), the rollout drifts behind and its samples grow
stale. Training on stale features hurts, so the consumer must *drop* them. This
module is the staleness half of the weight lifecycle (the hot-update / rollback /
serving accept-length gate are deliberately NOT here):

* ``WeightRegistry`` — the publish-ordered history of ``WeightVersion``s. Publish
  order IS the staleness axis: a sample's draft lag is its distance from the
  newest published version.
* ``StalenessPolicy`` — the **two axes** a sample can be stale on: the *draft*
  axis (how many published versions behind its draft weights are) and the
  *target* axis (was it produced against a now-superseded target model).
* ``DriftMonitor`` — rollout-distribution drift (the spread of draft lags), so
  the orchestrator can react when the pool falls behind.
* ``StalenessGatedQueue`` — wraps a ref queue so the consumer's loader only ever
  sees fresh refs; stale ones are acked (backpressure clears) and their features
  aborted (freed) instead of trained on.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from specforge.runtime.contracts import SampleRef, WeightVersion


class WeightRegistry:
    """Publish-ordered history of draft-weight versions (staleness axis).

    ``metadata_store`` (optional) makes the history durable across a restart;
    without it the history is in-process. Only the staleness surface lives here —
    no active-pointer / rollback / accept-length (that is the hot-update concern,
    out of scope).
    """

    def __init__(self, metadata_store: Optional[Any] = None) -> None:
        self._order: List[str] = []  # version_ids, oldest -> newest
        self._versions: Dict[str, WeightVersion] = {}
        self._store = metadata_store
        self._lock = threading.RLock()
        if self._store is not None and hasattr(self._store, "all_weight_versions"):
            for wv in self._store.all_weight_versions():
                self._order.append(wv.version_id)
                self._versions[wv.version_id] = wv

    def _persist(self, wv: WeightVersion) -> None:
        if self._store is not None and hasattr(self._store, "put_weight_version"):
            self._store.put_weight_version(wv)

    def publish(self, wv: WeightVersion) -> WeightVersion:
        """Append a new version (idempotent on version_id)."""
        with self._lock:
            if wv.version_id in self._versions:
                return self._versions[wv.version_id]
            self._versions[wv.version_id] = wv
            self._order.append(wv.version_id)
            self._persist(wv)
            return wv

    def latest(self) -> Optional[WeightVersion]:
        with self._lock:
            return self._versions[self._order[-1]] if self._order else None

    def get(self, version_id: str) -> Optional[WeightVersion]:
        with self._lock:
            return self._versions.get(version_id)

    def history(self) -> List[WeightVersion]:
        with self._lock:
            return [self._versions[v] for v in self._order]

    def draft_lag(self, draft_weight_version: Optional[str]) -> Optional[int]:
        """How many published versions are newer than this sample's draft weights.

        0 == the sample used the newest published draft; None == unknown version
        (treated as maximally stale by the policy).
        """
        with self._lock:
            for i in range(len(self._order) - 1, -1, -1):
                wv = self._versions[self._order[i]]
                if wv.draft_weight_version == draft_weight_version:
                    return (len(self._order) - 1) - i
            return None


@dataclass
class StalenessAssessment:
    draft_lag: Optional[int]
    target_stale: bool
    accept: bool
    reasons: List[str] = field(default_factory=list)


@dataclass
class StalenessPolicy:
    """Two independent staleness axes for a rollout sample.

    * **draft axis** — ``max_draft_lag``: reject a sample whose draft weights are
      more than N published versions behind (None = no draft bound).
    * **target axis** — ``require_target_match``: reject a sample produced against
      a target model version other than the current one. Unknown draft version =>
      maximally stale.
    """

    max_draft_lag: Optional[int] = None
    require_target_match: bool = True

    def assess(
        self,
        *,
        sample_draft_version: Optional[str],
        sample_target_version: str,
        registry: WeightRegistry,
        current_target_version: str,
    ) -> StalenessAssessment:
        reasons: List[str] = []
        lag = registry.draft_lag(sample_draft_version)
        accept = True
        if self.max_draft_lag is not None:
            if lag is None:
                accept = False
                reasons.append("unknown_draft_version")
            elif lag > self.max_draft_lag:
                accept = False
                reasons.append(f"draft_lag>{self.max_draft_lag}")
        target_stale = (
            self.require_target_match
            and sample_target_version != current_target_version
        )
        if target_stale:
            accept = False
            reasons.append("target_version_mismatch")
        return StalenessAssessment(
            draft_lag=lag, target_stale=target_stale, accept=accept, reasons=reasons
        )


class DriftMonitor:
    """Rollout-distribution drift: the spread of draft lags over recent samples.

    A healthy loop keeps lag ~0; a lagging or partially-updated pool shows a
    rising mean/max lag. ``drifting`` fires when the mean lag crosses a threshold
    so the orchestrator can react (pause, force a sync, alarm).
    """

    def __init__(self, window: int = 256) -> None:
        self._lags: Deque[int] = deque(maxlen=window)
        self._lock = threading.Lock()
        self._unknown = 0
        self._total = 0

    def observe(self, draft_lag: Optional[int]) -> None:
        with self._lock:
            self._total += 1
            if draft_lag is None:
                self._unknown += 1
            else:
                self._lags.append(int(draft_lag))

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            lags = list(self._lags)
            n = len(lags)
            return {
                "samples": self._total,
                "unknown_version": self._unknown,
                "window": n,
                "mean_lag": (sum(lags) / n) if n else 0.0,
                "max_lag": max(lags) if lags else 0,
            }

    def drifting(self, *, mean_lag_threshold: float) -> bool:
        return self.snapshot()["mean_lag"] > mean_lag_threshold


class StalenessGatedQueue:
    """Wrap a ref queue so the loader only sees fresh refs (online disagg gate).

    For each ref the inner queue yields, assess staleness; a fresh ref is passed
    through, a stale one is acked (so the producer's backpressure clears) and its
    features are ``abort``ed from the store (freed instead of trained on). Drift
    is recorded per assessed ref. Same get/ack/fail protocol the FeatureDataLoader
    consumes, so it drops in transparently.
    """

    def __init__(
        self,
        inner,
        *,
        feature_store,
        registry: WeightRegistry,
        policy: StalenessPolicy,
        current_target_version: str,
        drift: Optional[DriftMonitor] = None,
    ) -> None:
        self.inner = inner
        self.feature_store = feature_store
        self.registry = registry
        self.policy = policy
        self.current_target_version = current_target_version
        self.drift = drift
        self.dropped = 0

    def get(self, n: int, timeout_s: float = 0.0) -> List[SampleRef]:
        fresh: List[SampleRef] = []
        while len(fresh) < n:
            batch = self.inner.get(n - len(fresh), timeout_s=timeout_s)
            if not batch:  # channel closed-and-drained
                break
            for ref in batch:
                assessment = self.policy.assess(
                    sample_draft_version=ref.draft_weight_version,
                    sample_target_version=ref.target_model_version,
                    registry=self.registry,
                    current_target_version=self.current_target_version,
                )
                if self.drift is not None:
                    self.drift.observe(assessment.draft_lag)
                if assessment.accept:
                    fresh.append(ref)
                else:
                    # stale: free the features and ack so backpressure clears;
                    # the trainer never materializes it.
                    try:
                        self.feature_store.abort(
                            ref.sample_id, reason=",".join(assessment.reasons)
                        )
                    except Exception:  # pragma: no cover - best-effort free
                        pass
                    self.inner.ack([ref])
                    self.dropped += 1
        return fresh

    def ack(self, refs: List[SampleRef]) -> None:
        self.inner.ack(refs)

    def fail(self, refs: List[SampleRef], reason: str, retryable: bool) -> None:
        self.inner.fail(refs, reason, retryable)

    def depth(self) -> int:
        return self.inner.depth()


__all__ = [
    "WeightRegistry",
    "StalenessPolicy",
    "StalenessAssessment",
    "DriftMonitor",
    "StalenessGatedQueue",
]
