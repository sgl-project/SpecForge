# plan: promote train-with-decode to Phase 5

## Summary

Train-with-decode (W4) — one long-lived SGLang server simultaneously generating
training data and serving real spec-decoding traffic, with the trainer pushing
draft weights every N steps — is now treated as a real workload, not a Phase 6
"optional" item. The plan is updated end-to-end so the design primitives, phase
ordering, success criteria, and testing gates all reflect that.

The bet remains the same: **no Ray, no Mooncake**. W4 lands in Phase 5 with
~200 LOC additive on top of the Phase 4 `SGLangServerEngine` + `RemoteStream`
foundation. Actor topology and Mooncake transport stay behind explicit T1/T2/T3
triggers in §6.

## What changed in `plan.md`

| Section | Edit |
|---|---|
| Header | `Status` + `Last updated` bumped (2026-05-31). |
| §1 | New **"Workloads in scope"** subsection enumerating W1-W4 and noting W4 is the no-Ray bet. |
| §2.1 | New row in *Missing capabilities* for **Train-with-decode (W4)** with concrete code-level evidence. |
| §4 | New **§4.10 "Train-with-decode mode"** with the three primitives: `TargetEngine.update_draft_weights`, `Trainer._maybe_sync_draft_weights`, `ServingTrafficStream`, plus decode-mode `SGLangServerEngine` and serving-acceptance metrics. |
| §5 | New **Phase 5: Train-with-decode (Week 12-14)** with features #21-#24a. Old Phase 5 (Polish) → Phase 6 (Week 15-16), features renumbered #25-#30. Old Phase 6 (Optional) → Phase 7, features renumbered #31-#35. Old #29 "Train-with-decode" removed from Future. |
| §6 | *Ray + Mooncake vs HTTP/gRPC* tradeoff rewritten: explicit **T1 / T2 / T3 triggers** for when re-evaluation is warranted. T2 (multi-job sharing) is the only one that justifies adopting Ray. |
| §7 | New **Phase 5 plan** with 8 ordered steps (ABC update → server engine → trainer hook → traffic stream → metrics → e2e + correctness tests). |
| §8 | Removed *"No train-with-decode in Phase 1-5"*. Tightened other non-goals: explicit "exactly one engine per training job, including W4"; explicit "no inflight serving-request interception". |
| §9 | New Phase 5 success criterion (acceptance-rate slope > 0 across `weight_sync_interval` buckets; §10.5 gate passes). Old Phase 5 → Phase 6. |
| §10 | New **§10.5 weight-sync correctness gate** (4-step parity check: train 50 steps, push, compare logits + acceptance rate). New **§10.6 long-run weight-sync soak** for catching silent drift. |
| Cross-refs | Stale phase references chased: `Phase 5 #21` → `Phase 6 #25`, `Phase 6` (FSDP2) → `Phase 7 #31`, WSD deferral updated. |

## Why this matters

- W4 is the workload where the actor-topology vs. interface-extension judgment
  call actually fires. Putting it in writing forces the no-Ray claim to defend
  itself with concrete primitives (§4.10) and concrete triggers (§6 T1-T3),
  not vibes.
- The 4 ABCs from Phase 2 (`TargetEngine` / `HiddenStateStream` /
  `DraftTrainStrategy` / `Eagle3DraftModel`) are sufficient to absorb W4
  additively. If they weren't, this PR would have shown that and forced a
  topology rethink instead.
- §10.5 is new safety net: weight-sync silently no-op'ing is the most likely
  failure mode for W4 in production (HTTP returns 200, server keeps old
  weights, acceptance rate plateaus, no one notices for a week).

## Out of scope (explicit non-goals reaffirmed)

- Ray dependency
- Mooncake transport (gated behind T1/T3)
- Multi-job inference pool sharing (gated behind T2)
- Inflight serving-request interception (`ServingTrafficStream` reads from an
  external buffer the serving cluster owns)

## Test plan

- [ ] Re-read §4.10 and §5 Phase 5 with a fresh pair of eyes; confirm the
      three primitives + one optional stream cover what `_maybe_sync_draft_weights`
      in TorchSpec's `controller/loop.py:42-73` provides.
- [ ] Confirm phase numbering is consistent end-to-end (`grep -n "Phase \d"`).
- [ ] Sanity-check that §10.5's `atol/rtol` is reasonable across SGLang serving
      kernels vs trainer FSDP forward (1e-2 may be tight for some kernels).
- [ ] Once Phase 4 lands, this PR's primitives become the next tracked work item.
