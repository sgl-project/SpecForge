# coding=utf-8
"""DSpark draft-vocabulary pruning: buffers, objective, and run validation.

These run the real DSpark draft and online model on CPU with a tiny random
configuration, because the parts most likely to break silently -- which factor
of the Markov head follows which vocabulary, which id space the labels live in
-- are invisible to shape-only checks.
"""

import unittest
from collections import Counter
from unittest import mock

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
from specforge.algorithms.contracts import FeatureMode
from specforge.application.planning import _prunes_vocabulary, _validate_vocab_mapping
from specforge.config import Config
from specforge.data.preprocessing import process_token_dict_to_mappings
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.draft.vocab_mixin import OUT_OF_DRAFT_VOCAB_LABEL

VOCAB_SIZE = 256
DRAFT_VOCAB_SIZE = 64
HIDDEN = 32
BLOCK = 4
SEQ = 64
MARKOV_RANK = 8


def _draft_config(draft_vocab_size):
    config = Qwen3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
    )
    config.num_target_layers = 4
    config.block_size = BLOCK
    config.draft_vocab_size = draft_vocab_size
    config.layer_types = ["full_attention"] * 2
    config.dflash_config = {
        "projector_type": "dspark",
        "markov_rank": MARKOV_RANK,
        "markov_head_type": "vanilla",
        "target_layer_ids": [0, 1],
        "mask_token_id": 3,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
    }
    config._attn_implementation = "eager"
    return config


def _build(draft_vocab_size, seed=0):
    torch.manual_seed(seed)
    draft = DSparkDraftModel(_draft_config(draft_vocab_size))
    lm_head = nn.Linear(HIDDEN, VOCAB_SIZE, bias=False)
    torch.manual_seed(seed + 100)
    nn.init.normal_(lm_head.weight, std=0.02)
    lm_head.requires_grad_(False)
    model = OnlineDSparkModel(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=nn.Embedding(VOCAB_SIZE, HIDDEN),
        mask_token_id=3,
        block_size=BLOCK,
        attention_backend="eager",
        num_anchors=8,
        objective_chunk_blocks=4,
    )
    return draft, model


def _inputs(seed=5):
    torch.manual_seed(seed)
    return {
        "input_ids": torch.randint(0, VOCAB_SIZE, (1, SEQ)),
        "hidden_states": torch.randn(1, SEQ, 2 * HIDDEN),
        "loss_mask": torch.ones(1, SEQ),
        "target_last_hidden_states": torch.randn(1, SEQ, HIDDEN),
    }


def _mapping(draft_vocab_size=DRAFT_VOCAB_SIZE):
    """Frequencies whose top-K is scattered, so d2t is a non-trivial offset table.

    A contiguous selection would make d2t all zeros and hide any ordering bug.
    """
    counts = Counter(
        {
            token: (VOCAB_SIZE - token if token % 2 == 0 else 1)
            for token in range(VOCAB_SIZE)
        }
    )
    d2t, t2d = process_token_dict_to_mappings(counts, draft_vocab_size, VOCAB_SIZE)
    return t2d, d2t


#: Names for ``_dspark_objective_chunk_terms``'s positional return, so the
#: reference below can be compared by meaning instead of by index.
_TERM_NAMES = (
    "ce_num",
    "l1_num",
    "confidence_num",
    "confidence_error_num",
    "correct_num",
    "eval_den",
    "ce_eval_den",
    "ce_position_num",
    "ce_position_den",
    "correct_position_num",
    "position_den",
    "teacher_agreement_num",
    "teacher_top1_num",
    "draft_top1_num",
    "tau_num",
    "tau_den",
    "kept_mass_num",
)


def _chunk_term_args(model, blocks=4, seed=11):
    """Build one chunk's worth of objective inputs, shaped as the forward does."""
    torch.manual_seed(seed)
    hidden = torch.randn(1, blocks, BLOCK, HIDDEN)
    prev_token_ids = torch.randint(0, VOCAB_SIZE, (1, blocks, BLOCK))
    target_ids = torch.randint(0, VOCAB_SIZE, (1, blocks, BLOCK))
    # eval_mask is a per-block prefix mask in the real forward (cumprod), so
    # reproduce that shape of truth rather than an arbitrary boolean field.
    keep = torch.tensor([[[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]]])
    eval_mask = keep[:, :blocks].bool()
    loss_weights = eval_mask.to(torch.float32)
    aligned_target_hidden = torch.randn(1, blocks, BLOCK, HIDDEN)
    return (
        hidden,
        prev_token_ids,
        target_ids,
        loss_weights,
        eval_mask,
        aligned_target_hidden,
    )


def _reference_dspark_chunk_terms(
    model,
    hidden,
    prev_token_ids,
    target_ids,
    loss_weights,
    eval_mask,
    aligned_target_hidden,
):
    """The DSpark objective exactly as it stood at 7712377, before pruning.

    Transcribed rather than imported on purpose: a reference that is generated
    from the implementation cannot detect a change in the implementation. Keep
    this function frozen -- if a deliberate change to the unpruned objective is
    ever made, the right move is to update it in the same commit and say so,
    not to relax the comparison.
    """
    batch_size, num_blocks, block_size, hidden_size = hidden.shape
    base_logits = model.lm_head(
        hidden.reshape(batch_size, num_blocks * block_size, hidden_size)
    ).reshape(batch_size, num_blocks, block_size, -1)
    draft_logits = model.draft_model.apply_logits_head(
        base_logits,
        prev_token_ids=prev_token_ids,
        hidden_states=hidden,
    )
    vocab_size = draft_logits.shape[-1]
    cross_entropy = torch.nn.functional.cross_entropy(
        draft_logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape_as(target_ids)
    terms = {"ce_num": (cross_entropy * loss_weights).sum()}

    with torch.no_grad():
        target_logits = model.lm_head(
            aligned_target_hidden.reshape(
                batch_size, num_blocks * block_size, hidden_size
            )
        ).reshape_as(draft_logits)
        target_probabilities = torch.softmax(target_logits.float(), dim=-1)
        teacher_ids = target_logits.argmax(dim=-1)
    draft_probabilities = torch.softmax(draft_logits.float(), dim=-1)
    l1_per_token = (draft_probabilities - target_probabilities).abs().sum(dim=-1)
    accept_probability = (1.0 - 0.5 * l1_per_token).clamp(0.0, 1.0)
    terms["l1_num"] = (l1_per_token * loss_weights).sum()

    confidence_pred = model.draft_model.predict_confidence(
        hidden, prev_token_ids=prev_token_ids
    )
    confidence_per_token = torch.nn.functional.binary_cross_entropy_with_logits(
        confidence_pred.float(),
        accept_probability.detach(),
        reduction="none",
    )
    terms["confidence_num"] = (confidence_per_token * loss_weights).sum()
    terms["confidence_error_num"] = (
        (confidence_pred.float().sigmoid() - accept_probability).abs() * loss_weights
    ).sum()

    with torch.no_grad():
        predicted_ids = draft_logits.argmax(dim=-1)
        correct = ((predicted_ids == target_ids) & eval_mask).float()
        terms["correct_num"] = correct.sum()
        terms["eval_den"] = eval_mask.float().sum()
        terms["ce_position_num"] = (cross_entropy.detach() * eval_mask).sum(dim=(0, 1))
        terms["correct_position_num"] = correct.sum(dim=(0, 1))
        terms["position_den"] = eval_mask.float().sum(dim=(0, 1))
        terms["teacher_agreement_num"] = (
            (predicted_ids == teacher_ids).float() * eval_mask
        ).sum()
        terms["teacher_top1_num"] = (
            target_probabilities.max(dim=-1).values * eval_mask
        ).sum()
        terms["draft_top1_num"] = (
            draft_probabilities.max(dim=-1).values * eval_mask
        ).sum()
        valid_blocks = eval_mask.any(dim=-1).float()
        accepted_expectation = (accept_probability.detach() * eval_mask).cumprod(
            dim=-1
        ).sum(dim=-1) + 1.0
        terms["tau_num"] = (accepted_expectation * valid_blocks).sum()
        terms["tau_den"] = valid_blocks.sum()
    return terms


class DSparkDraftVocabTest(unittest.TestCase):
    def test_full_vocab_draft_keeps_an_unchanged_state_dict(self):
        """Existing checkpoints must stay loadable: no new keys when unpruned."""
        full, _ = _build(VOCAB_SIZE)
        pruned, _ = _build(DRAFT_VOCAB_SIZE)

        self.assertFalse(full.use_draft_vocab)
        self.assertNotIn("t2d", full.state_dict())
        self.assertNotIn("d2t", full.state_dict())

        self.assertTrue(pruned.use_draft_vocab)
        self.assertIn("t2d", pruned.state_dict())
        self.assertIn("d2t", pruned.state_dict())

    def test_markov_head_conditions_on_full_vocab_and_emits_draft_vocab(self):
        """W1 reads the previous real token; only W2 follows the pruned vocab."""
        pruned, _ = _build(DRAFT_VOCAB_SIZE)
        head = pruned.markov_head

        self.assertEqual(tuple(head.markov_w1.weight.shape), (VOCAB_SIZE, MARKOV_RANK))
        self.assertEqual(
            tuple(head.markov_w2.weight.shape), (DRAFT_VOCAB_SIZE, MARKOV_RANK)
        )

    def test_forward_before_the_mapping_is_installed_raises(self):
        """Zeroed buffers are silently wrong, so using them must be impossible."""
        _pruned, model = _build(DRAFT_VOCAB_SIZE)
        with self.assertRaisesRegex(RuntimeError, "no t2d/d2t"):
            model(**_inputs())

    def test_full_vocab_objective_matches_the_pre_pruning_formula(self):
        """Pin the unpruned objective against the formula it had before pruning.

        Comparing two runs of the *current* implementation would only prove
        determinism -- a regression that moved both sides equally would still
        pass. ``_reference_dspark_chunk_terms`` is a transcription of the
        objective at 7712377, so it cannot drift along with the code under test.
        """
        _draft, model = _build(VOCAB_SIZE, seed=1)
        args = _chunk_term_args(model)

        expected = _reference_dspark_chunk_terms(model, *args)
        # The current signature splits CE weights out of the shared weights; for
        # a full-vocabulary run the two are the same tensor by construction.
        hidden, prev_token_ids, target_ids, loss_weights, eval_mask, teacher = args
        actual = dict(
            zip(
                _TERM_NAMES,
                model._dspark_objective_chunk_terms(
                    hidden,
                    prev_token_ids,
                    target_ids,
                    loss_weights,
                    loss_weights,
                    eval_mask,
                    teacher,
                ),
            )
        )

        self.assertEqual(len(actual), len(_TERM_NAMES))
        self.assertTrue(set(expected) <= set(actual))
        for name, want in expected.items():
            with self.subTest(term=name):
                # Bit-exact: the claim is that the unpruned path is the same
                # arithmetic, not merely a close approximation of it.
                self.assertTrue(
                    torch.equal(actual[name], want),
                    f"{name}: {actual[name]} != {want}",
                )

    def test_full_vocab_loss_keeps_the_single_denominator_form(self):
        """Pin the final scalar too, not just the terms it is built from.

        Pruning gives cross-entropy its own denominator. Unpruned the two
        denominators are equal, so the split form is algebraically the same --
        but ``x/D + y/D`` and ``(x + y)/D`` do not round the same way once a
        decayed ``loss_weights`` makes ``D`` inexact, which is precisely the
        configuration this checks. Existing runs must not see their loss move.
        """
        for gamma in (None, 3.0):
            with self.subTest(loss_decay_gamma=gamma):
                _draft, model = _build(VOCAB_SIZE, seed=1)
                model.loss_decay_gamma = gamma
                torch.manual_seed(42)
                loss, _accuracy, _metrics = model(**_inputs())

                # The metrics carry the very tensors the loss was built from, so
                # the pre-pruning expression can be rebuilt exactly.
                ratios = _metrics["ratio_metrics"]
                ce_num, _ce_den = ratios["ce_loss"]
                l1_num, loss_den = ratios["l1_loss"]
                confidence_num, _ = ratios["confidence_loss"]
                world_size = 1  # no process group is initialized under test
                expected = (
                    world_size
                    * (
                        model.dspark_ce_loss_alpha * ce_num
                        + model.dspark_l1_loss_alpha * l1_num
                        + model.dspark_confidence_head_alpha * confidence_num
                    )
                    / loss_den
                )
                self.assertTrue(torch.equal(loss, expected), f"{loss} != {expected}")

    def test_full_vocab_run_emits_one_denominator_all_reduce(self):
        """Pin the collective sequence: single-process runs cannot observe it.

        Ranks must agree on the count and order of collectives. Pruning gives
        cross-entropy its own denominator and therefore a second all_reduce; an
        unpruned run must keep emitting exactly the one it always did, or a
        multi-GPU resume onto this branch would desynchronize.
        """
        for draft_vocab_size, expected in ((VOCAB_SIZE, 1), (DRAFT_VOCAB_SIZE, 2)):
            with self.subTest(draft_vocab_size=draft_vocab_size):
                draft, model = _build(draft_vocab_size)
                if draft_vocab_size != VOCAB_SIZE:
                    draft.install_vocab_mapping(*_mapping())
                calls = []
                with mock.patch.multiple(
                    "torch.distributed",
                    is_available=lambda: True,
                    is_initialized=lambda: True,
                    get_world_size=lambda: 4,
                    all_reduce=lambda tensor, **kw: calls.append(tensor),
                ):
                    model(**_inputs())
                self.assertEqual(len(calls), expected)

    def test_full_vocab_coverage_terms_are_degenerate(self):
        """The two terms pruning added must be no-ops without pruning."""
        _draft, model = _build(VOCAB_SIZE, seed=1)
        hidden, prev_token_ids, target_ids, loss_weights, eval_mask, teacher = (
            _chunk_term_args(model)
        )

        terms = model._dspark_objective_chunk_terms(
            hidden,
            prev_token_ids,
            target_ids,
            loss_weights,
            loss_weights,
            eval_mask,
            teacher,
        )
        named = dict(zip(_TERM_NAMES, terms))
        ce_eval_den = named["ce_eval_den"]
        ce_position_den = named["ce_position_den"]

        # Every label is in vocabulary, so the CE evaluation set is the full one.
        self.assertTrue(torch.equal(ce_eval_den, eval_mask.float().sum()))
        self.assertTrue(torch.equal(ce_position_den, eval_mask.float().sum(dim=(0, 1))))

    def test_pruned_objective_trains_and_reports_coverage(self):
        pruned, model = _build(DRAFT_VOCAB_SIZE)
        t2d, d2t = _mapping()
        pruned.install_vocab_mapping(t2d, d2t)

        loss, _accuracy, metrics = model(**_inputs())
        loss.backward()

        ratios = metrics["ratio_metrics"]
        self.assertIn("draft_vocab_coverage", ratios)
        numerator, denominator = ratios["draft_vocab_coverage"]
        coverage = float(numerator) / float(denominator)
        self.assertGreater(coverage, 0.0)
        self.assertLessEqual(coverage, 1.0)

        # The pruned factor must actually receive gradient.
        self.assertIsNotNone(pruned.markov_head.markov_w2.weight.grad)
        self.assertGreater(float(pruned.markov_head.markov_w2.weight.grad.norm()), 0.0)

    def test_cross_entropy_denominator_excludes_pruned_labels(self):
        """ce_loss is a mean over in-vocabulary positions, not over all of them."""
        pruned, model = _build(DRAFT_VOCAB_SIZE)
        t2d, d2t = _mapping()
        pruned.install_vocab_mapping(t2d, d2t)

        _loss, _accuracy, metrics = model(**_inputs())
        ratios = metrics["ratio_metrics"]
        ce_denominator = float(ratios["ce_loss"][1])
        l1_denominator = float(ratios["l1_loss"][1])
        coverage_numerator = float(ratios["draft_vocab_coverage"][0])

        self.assertLess(ce_denominator, l1_denominator)
        self.assertAlmostEqual(ce_denominator, coverage_numerator, places=4)

    def test_label_lookup_maps_kept_tokens_and_flags_pruned_ones(self):
        pruned, _model = _build(DRAFT_VOCAB_SIZE)
        t2d, d2t = _mapping()
        pruned.install_vocab_mapping(t2d, d2t)

        index = pruned.draft_vocab_index()
        kept = torch.nonzero(t2d.bool()).flatten()

        self.assertTrue(
            torch.equal(index[kept], torch.arange(DRAFT_VOCAB_SIZE, dtype=torch.long))
        )
        dropped = torch.nonzero(~t2d.bool()).flatten()
        if dropped.numel():
            self.assertTrue(bool((index[dropped] == OUT_OF_DRAFT_VOCAB_LABEL).all()))

    def test_draft_ids_round_trip_to_the_target_vocabulary(self):
        """Generation feeds sampled ids back into W1, so they must be target ids."""
        pruned, _model = _build(DRAFT_VOCAB_SIZE)
        t2d, d2t = _mapping()
        pruned.install_vocab_mapping(t2d, d2t)

        target_ids = pruned.draft_ids_to_target_ids(torch.arange(DRAFT_VOCAB_SIZE))
        self.assertTrue(torch.equal(target_ids, torch.nonzero(t2d.bool()).flatten()))
        self.assertLess(int(target_ids.max()), VOCAB_SIZE)

    def test_install_rejects_an_inconsistent_mapping(self):
        pruned, _model = _build(DRAFT_VOCAB_SIZE)
        t2d, d2t = _mapping()
        with self.assertRaises(ValueError):
            pruned.install_vocab_mapping(t2d, d2t.flip(0))

    def test_reinstalling_a_mapping_invalidates_the_cached_head(self):
        pruned, model = _build(DRAFT_VOCAB_SIZE)
        t2d, d2t = _mapping()
        pruned.install_vocab_mapping(t2d, d2t)
        first = model._pruned_head_state()[0].clone()

        other_counts = Counter({token: token + 1 for token in range(VOCAB_SIZE)})
        other_d2t, other_t2d = process_token_dict_to_mappings(
            other_counts, DRAFT_VOCAB_SIZE, VOCAB_SIZE
        )
        pruned.install_vocab_mapping(other_t2d, other_d2t)
        second = model._pruned_head_state()[0]

        self.assertFalse(torch.equal(first, second))

    def test_mapping_survives_a_state_dict_round_trip(self):
        """A reloaded pruned checkpoint must be runnable, not just byte-correct.

        t2d/d2t travel in the state dict, but load_state_dict calls none of our
        methods; if "is a mapping installed" were tracked as a flag, a correctly
        saved checkpoint would reload and then refuse to run.
        """
        source, _model = _build(DRAFT_VOCAB_SIZE)
        t2d, d2t = _mapping()
        source.install_vocab_mapping(t2d, d2t)

        target, target_model = _build(DRAFT_VOCAB_SIZE, seed=7)
        self.assertFalse(target.vocab_mapping_loaded)
        target.load_state_dict(source.state_dict())

        self.assertTrue(target.vocab_mapping_loaded)
        self.assertTrue(torch.equal(target.t2d, source.t2d))
        self.assertTrue(torch.equal(target.d2t, source.d2t))
        self.assertTrue(
            torch.equal(target.draft_vocab_index(), source.draft_vocab_index())
        )
        target_model(**_inputs())

    def test_full_vocab_logsumexp_matches_the_dense_computation(self):
        """The chunked normalizer must equal the one-shot one, exactly enough.

        It is walked in vocabulary chunks to avoid materializing [.., V]; that
        partitioning is the only thing that could make it wrong.
        """
        _draft, model = _build(DRAFT_VOCAB_SIZE)
        hidden = torch.randn(1, 12, HIDDEN)

        chunked = model._full_vocab_logsumexp(hidden)
        dense = torch.logsumexp(model.lm_head(hidden).float(), dim=-1)

        self.assertEqual(chunked.shape, dense.shape)
        self.assertTrue(torch.allclose(chunked, dense, atol=1e-6, rtol=1e-6))

    def test_full_vocab_logsumexp_is_chunk_boundary_independent(self):
        """Chunking is an implementation detail and must not change the value."""
        import specforge.algorithms.common.dflash_family_model as module

        _draft, model = _build(DRAFT_VOCAB_SIZE)
        hidden = torch.randn(1, 8, HIDDEN)
        original = module._TEACHER_NORMALIZER_VOCAB_CHUNK
        values = []
        try:
            for chunk in (7, 64, VOCAB_SIZE, VOCAB_SIZE * 2):
                module._TEACHER_NORMALIZER_VOCAB_CHUNK = chunk
                values.append(model._full_vocab_logsumexp(hidden))
        finally:
            module._TEACHER_NORMALIZER_VOCAB_CHUNK = original

        for value in values[1:]:
            self.assertTrue(torch.allclose(values[0], value, atol=1e-6, rtol=1e-6))

    def test_acceptance_uses_true_target_mass_not_the_conditional(self):
        """The review's counterexample, made analytic.

        Build a teacher that puts all its mass uniformly on four tokens, two of
        which the draft vocabulary keeps. The reachable mass is then 1/2, and
        acceptance can never exceed that. Renormalizing over the kept tokens --
        what the pruned path used to do -- would report a teacher summing to 1
        and an acceptance ceiling of 1, hiding exactly the cost of pruning.
        """
        pruned, model = _build(DRAFT_VOCAB_SIZE)
        t2d, d2t = _mapping()
        pruned.install_vocab_mapping(t2d, d2t)

        kept = torch.nonzero(t2d, as_tuple=False).flatten().tolist()
        dropped = [t for t in range(VOCAB_SIZE) if t not in set(kept)]
        favoured = [kept[0], kept[1], dropped[0], dropped[1]]

        # logit_v = weight[v, 0] once the teacher state is the first basis
        # vector, so the target distribution is set directly.
        with torch.no_grad():
            model.lm_head.weight.fill_(0.0)
            model.lm_head.weight[:, 0] = -50.0
            for token in favoured:
                model.lm_head.weight[token, 0] = 0.0

        batch = _inputs()
        teacher_state = torch.zeros(1, SEQ, HIDDEN)
        teacher_state[:, :, 0] = 1.0
        batch["target_last_hidden_states"] = teacher_state

        _loss, _accuracy, metrics = model(**batch)
        ratios = metrics["ratio_metrics"]

        kept_mass_num, eval_den = ratios["teacher_kept_mass"]
        kept_mass = float(kept_mass_num) / float(eval_den)
        self.assertAlmostEqual(kept_mass, 0.5, places=3)

        # kept_mass bounds each SINGLE-STEP acceptance probability, not tau.
        # tau = 1 + sum_j prod_{k<=j} a_k, so the bound below is the loose one
        # that follows from prod_{k<=j} a_k <= a_j <= kept_mass: an anchor token
        # plus at most one kept_mass per suffix position.
        tau_num, tau_den = ratios["tau_probabilistic"]
        tau = float(tau_num) / float(tau_den)
        self.assertLessEqual(tau, 1.0 + BLOCK * kept_mass + 1e-4)

    def test_full_vocab_run_reports_no_kept_mass(self):
        """Without pruning there is nothing to lose, so the metric is absent."""
        _draft, model = _build(VOCAB_SIZE)
        _loss, _accuracy, metrics = model(**_inputs())
        self.assertNotIn("teacher_kept_mass", metrics["ratio_metrics"])

    def test_loading_a_conflicting_mapping_is_rejected(self):
        """The resume hazard: same K, different tokens, no error, wrong labels.

        Nothing about the shapes disagrees, so without this check the run keeps
        training with lm_head rows that mean different tokens than it believes.
        """
        resolved, _model = _build(DRAFT_VOCAB_SIZE)
        t2d, d2t = _mapping()
        resolved.install_vocab_mapping(t2d, d2t)

        other = Counter({token: token + 1 for token in range(VOCAB_SIZE)})
        other_d2t, other_t2d = process_token_dict_to_mappings(
            other, DRAFT_VOCAB_SIZE, VOCAB_SIZE
        )
        donor, _ = _build(DRAFT_VOCAB_SIZE, seed=7)
        donor.install_vocab_mapping(other_t2d, other_d2t)
        self.assertEqual(int(donor.t2d.sum()), int(resolved.t2d.sum()))

        with self.assertRaisesRegex(ValueError, "differs from the one this run"):
            resolved.load_state_dict(donor.state_dict())

    def test_loading_an_identical_mapping_is_allowed(self):
        """A matching mapping is the normal resume; it must not be an error."""
        resolved, _model = _build(DRAFT_VOCAB_SIZE)
        donor, _ = _build(DRAFT_VOCAB_SIZE, seed=7)
        t2d, d2t = _mapping()
        resolved.install_vocab_mapping(t2d, d2t)
        donor.install_vocab_mapping(t2d, d2t)

        resolved.load_state_dict(donor.state_dict())
        self.assertTrue(torch.equal(resolved.t2d, t2d))

    def test_draft_vocab_size_rejects_degenerate_values(self):
        """0 must not be coerced to "full vocabulary" behind the user's back."""
        for bad in (0, -1, True, 2.5, VOCAB_SIZE + 1):
            with self.subTest(draft_vocab_size=bad):
                with self.assertRaises(ValueError):
                    DSparkDraftModel(_draft_config(bad))

        # None remains the only spelling of "unset".
        unset = DSparkDraftModel(_draft_config(None))
        self.assertEqual(unset.draft_vocab_size, VOCAB_SIZE)
        self.assertFalse(unset.use_draft_vocab)


class DSparkResumeContractTest(unittest.TestCase):
    """The resume contract must not grow a key unpruned runs cannot supply."""

    STEP = builtin_algorithm_registry().resolve("dspark").providers.step

    def _contract(self, draft_vocab_size):
        draft, model = _build(draft_vocab_size)
        return self.STEP.resume_contract(None, draft, model)

    def test_full_vocab_contract_has_no_draft_vocab_size_key(self):
        """Trainer treats a contract key a checkpoint lacks as fatal.

        Every DSpark checkpoint written before pruning existed carries no
        dspark_draft_vocab_size, so recording it unconditionally would make all
        of them unresumable -- for a field that, unpruned, could only ever hold
        vocab_size.
        """
        self.assertNotIn("dspark_draft_vocab_size", self._contract(VOCAB_SIZE))

    def test_pruned_contract_records_the_draft_vocab_size(self):
        contract = self._contract(DRAFT_VOCAB_SIZE)
        self.assertEqual(contract["dspark_draft_vocab_size"], DRAFT_VOCAB_SIZE)

    def test_pruning_adds_exactly_one_key(self):
        """Nothing else about the contract may shift with the feature."""
        full = self._contract(VOCAB_SIZE)
        pruned = self._contract(DRAFT_VOCAB_SIZE)
        self.assertEqual(set(pruned) - set(full), {"dspark_draft_vocab_size"})
        self.assertEqual(set(full) - set(pruned), set())


class DSparkVocabMappingPlanningTest(unittest.TestCase):
    """Requiring an explicit mapping must key off pruning, not capability."""

    ALGORITHM = builtin_algorithm_registry().resolve("dspark")

    def _config(self, draft_config, **model_overrides):
        model = {
            "target_model_path": "target/model",
            "draft_model_config": draft_config,
            "target_backend": "sglang",
        }
        model.update(model_overrides)
        return Config.model_validate(
            {
                "model": model,
                "data": {"train_data_path": "train.jsonl"},
                "training": {"strategy": "dspark", "max_steps": 1},
                "deployment": {
                    "mode": "disaggregated",
                    "disaggregated": {
                        "control_dir": "/control",
                        "backend": "mooncake",
                        "server_urls": ["http://capture.invalid:30000"],
                    },
                },
            }
        )

    def test_full_vocab_disaggregated_run_needs_no_mapping_path(self):
        cfg = self._config("configs/qwen3-8b-dspark.json")
        self.assertFalse(_prunes_vocabulary(cfg, self.ALGORITHM))
        _validate_vocab_mapping(cfg, self.ALGORITHM, FeatureMode.STREAMING)

    def test_unreadable_draft_config_is_unknown_not_unpruned(self):
        """An unreadable draft config answers "unknown", never "unpruned".

        Config validation must keep working without the draft config on disk, so
        this cannot resolve it. Each rule then decides what unknown means for
        itself: DSpark carries t2d/d2t only when it prunes, so an unproven run
        is not held to the shared-mapping requirement, and the real resolution
        error is reported at the model-loading boundary. EAGLE3, which always
        carries the buffers, keeps its unconditional requirement -- covered by
        tests/test_config/test_schema.py.
        """
        cfg = self._config("missing-draft-config")
        with mock.patch(
            "specforge.training.model_loading.draft_config_dict",
            side_effect=OSError("draft config is unavailable"),
        ):
            self.assertIsNone(_prunes_vocabulary(cfg, self.ALGORITHM))
            self.assertFalse(
                self.ALGORITHM.spec.capabilities.keeps_vocab_buffers_when_unpruned
            )
            _validate_vocab_mapping(cfg, self.ALGORITHM, FeatureMode.STREAMING)

    def test_unreadable_draft_config_does_not_reject_a_mapping_path(self):
        """Only a config proven unpruned may be told to drop its mapping path.

        Rejecting on "unknown" would answer an unreadable draft config with a
        confident, wrong instruction to delete a correct setting; the real
        resolution error is reported at the model-loading boundary instead.
        """
        cfg = self._config(
            "missing-draft-config",
            vocab_mapping_path="mapping.pt",
        )
        with mock.patch(
            "specforge.training.model_loading.draft_config_dict",
            side_effect=OSError("draft config is unavailable"),
        ):
            _validate_vocab_mapping(cfg, self.ALGORITHM, FeatureMode.STREAMING)

    def test_pruned_disaggregated_run_requires_a_mapping_path(self):
        cfg = self._config("configs/qwen3-8b-dspark-draftvocab32k.json")
        self.assertTrue(_prunes_vocabulary(cfg, self.ALGORITHM))
        with self.assertRaisesRegex(ValueError, "vocab_mapping_path"):
            _validate_vocab_mapping(cfg, self.ALGORITHM, FeatureMode.STREAMING)

    def test_mapping_path_on_a_full_vocab_run_is_rejected(self):
        """Capability is not use: a full-vocab run has nothing to map.

        Without this the path is accepted here and then fails deep in model
        construction with an error about missing t2d/d2t buffers, which points
        at the model instead of at the config that is actually wrong.
        """
        cfg = self._config(
            "configs/qwen3-8b-dspark.json", vocab_mapping_path="mapping.pt"
        )
        with self.assertRaisesRegex(ValueError, "no t2d/d2t buffers"):
            _validate_vocab_mapping(cfg, self.ALGORITHM, FeatureMode.STREAMING)

    def test_mapping_path_on_an_incapable_algorithm_is_rejected(self):
        dflash = builtin_algorithm_registry().resolve("dflash")
        cfg = Config.model_validate(
            {
                "model": {
                    "target_model_path": "target/model",
                    "draft_model_config": "configs/qwen3-8b-dflash.json",
                    "vocab_mapping_path": "mapping.pt",
                },
                "data": {"hidden_states_path": "features"},
                "training": {"strategy": "dflash"},
            }
        )
        with self.assertRaisesRegex(ValueError, "does not support vocabulary mapping"):
            _validate_vocab_mapping(cfg, dflash, FeatureMode.OFFLINE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
