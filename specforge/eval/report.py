"""A compact, tracker-ready view of teacher-forced checkpoint diagnostics."""

from __future__ import annotations

import math
import re

_SUMMARY_NAMES = {
    "loss": "sample_mean_loss",
    "ce_loss": "ce_loss",
    "lk_loss": "lk_loss",
    "dflash2/selector/loss": "selector_loss",
    "dflash/hard_label/unary_greedy_accepted_length": "unary_greedy_accepted_length",
    "dflash2/selector/greedy_accepted_length": "selector_greedy_accepted_length",
}
_POSITION_NAMES = {
    "hard_label/unary_top1_accuracy": "unary_top1_accuracy",
    "hard_label/unary_greedy_prefix_acceptance": "unary_prefix_acceptance",
    "hard_label/unary_greedy_prefix_survival": "unary_prefix_survival",
    "selector/greedy_prefix_acceptance": "selector_prefix_acceptance",
    "selector/greedy_prefix_survival": "selector_prefix_survival",
    "selector/greedy_prefix_covered_accuracy": "selector_prefix_covered_accuracy",
    "selector/greedy_prefix_coverage_miss_rate": "selector_prefix_coverage_miss_rate",
    "selector/greedy_prefix_ranking_error_rate": "selector_prefix_ranking_error_rate",
}


def checkpoint_metrics(
    report: dict, *, include_diagnostics: bool = False
) -> dict[str, float]:
    """Return summary/position groups while leaving the raw report untouched."""
    if not report.get("success") or report.get("examples", 0) <= 0:
        raise ValueError("expected a successful, nonempty checkpoint evaluation report")
    grouped = {}
    for name, scalar in report["mean"].items():
        value = float(scalar)
        if not math.isfinite(value):
            raise ValueError(f"non-finite metric {name}: {value}")
        pair = report.get("ratio_totals", {}).get(name)
        if pair is not None:
            numerator, denominator = map(float, pair)
            if (
                not math.isfinite(numerator)
                or not math.isfinite(denominator)
                or denominator < 0
            ):
                raise ValueError(f"invalid ratio totals for {name}: {pair}")
            if denominator == 0:
                if numerator != 0:
                    raise ValueError(
                        f"invalid zero-denominator ratio for {name}: {pair}"
                    )
                continue
            if not math.isclose(
                value, numerator / denominator, rel_tol=1e-6, abs_tol=1e-9
            ):
                raise ValueError(f"metric {name} disagrees with its pooled counts")
        if name in _SUMMARY_NAMES:
            key = f"eval/summary/{_SUMMARY_NAMES[name]}"
        elif match := re.fullmatch(
            r"dflash/hard_label/unary_top(\d+)_oracle_accepted_length", name
        ):
            key = f"eval/oracle/top{match[1]}_accepted_length"
        elif match := re.fullmatch(r"position_([1-9]\d*)/(.+)", name):
            position, metric = match.groups()
            label = _POSITION_NAMES.get(metric)
            if label is None:
                if oracle := re.fullmatch(
                    r"hard_label/unary_top(\d+)_oracle_prefix_(acceptance|survival)",
                    metric,
                ):
                    label = f"oracle_top{oracle[1]}_prefix_{oracle[2]}"
                elif include_diagnostics:
                    label = f"diagnostics/{metric}"
                else:
                    continue
            key = f"eval/position_{position}/{label}"
        elif include_diagnostics:
            key = f"eval/diagnostics/{name}"
        else:
            continue
        if key in grouped:
            raise ValueError(f"multiple raw metrics map to {key}")
        grouped[key] = value
    if include_diagnostics:
        for name, scalar in report.get("sum_totals", {}).items():
            value = float(scalar)
            if not math.isfinite(value):
                raise ValueError(f"non-finite count {name}: {value}")
            if match := re.fullmatch(r"position_([1-9]\d*)/(.+)", name):
                key = f"eval/position_{match[1]}/counts/{match[2]}"
            else:
                key = f"eval/counts/{name}"
            if key in grouped:
                raise ValueError(f"multiple raw metrics map to {key}")
            grouped[key] = value
    return grouped


def render_checkpoint_report(report: dict) -> str:
    """Show overall loss/length first, followed by one row per draft position."""
    metrics = checkpoint_metrics(report)
    lines = [
        "# Teacher-forced checkpoint evaluation",
        "",
        f"Examples: **{report['examples']}**. Saved-continuation diagnostics; serving acceptance, speed and answer accuracy require generation benchmarks.",
        "",
        "| Summary | Value |",
        "| --- | ---: |",
    ]
    labels = {
        "sample_mean_loss": "Composite loss (mean per example)",
        "ce_loss": "CE loss (pooled)",
        "lk_loss": "LK loss (pooled)",
        "selector_loss": "Selector loss (pooled)",
        "unary_greedy_accepted_length": "Unary greedy length (proxy)",
        "selector_greedy_accepted_length": "Selector greedy length (proxy)",
    }
    for name, label in labels.items():
        if (value := metrics.get(f"eval/summary/{name}")) is not None:
            lines.append(f"| {label} | {value:.4f} |")
    for key, value in metrics.items():
        if key.startswith("eval/oracle/"):
            top_k = key.split("/")[-1].split("_")[0]
            lines.append(
                f"| {top_k} oracle length (candidate-coverage bound) | {value:.4f} |"
            )
    positions = sorted(
        {
            int(match[1])
            for key in report["mean"]
            if (match := re.match(r"position_([1-9]\d*)/", key))
        }
    )
    lines += [
        "",
        "Lengths include one anchor token plus the surviving proposal prefix.",
        "Composite and component losses use different averaging denominators; the displayed components need not sum to the composite.",
        "",
    ]
    if positions:
        lines += [
            "| Position | Unary top-1 | Unary conditional | Unary survival | Selector conditional | Selector survival |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        columns = (
            "unary_top1_accuracy",
            "unary_prefix_acceptance",
            "unary_prefix_survival",
            "selector_prefix_acceptance",
            "selector_prefix_survival",
        )
        for position in positions:
            values = [
                metrics.get(f"eval/position_{position}/{column}") for column in columns
            ]
            cells = [
                "—" if value is None else f"{100 * value:.2f}%" for value in values
            ]
            lines.append(f"| {position} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "Conditional = accepted / reached prefixes. Survival = accepted / total evaluated blocks (including censored tails). Unary top-1 is marginal accuracy. Zero-exposure metrics are omitted (—); inspect ratio_totals for denominators.",
            "",
        ]
    return "\n".join(lines)
