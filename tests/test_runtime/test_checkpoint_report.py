"""Presentation must not conflate marginal, conditional, survival or oracle metrics."""

import copy
import json

import pytest

from specforge.cli import main
from specforge.eval.report import checkpoint_metrics, render_checkpoint_report


@pytest.fixture
def report():
    return {
        "success": True,
        "examples": 2,
        "mean": {
            "loss": 2.5,
            "ce_loss": 2.0,
            "dflash2/selector/greedy_accepted_length": 2.25,
            "dflash/hard_label/unary_top16_oracle_accepted_length": 3.5,
            "position_1/selector/greedy_prefix_acceptance": 0.75,
            "position_1/selector/greedy_prefix_survival": 0.75,
            "position_2/selector/greedy_prefix_acceptance": 2 / 3,
            "position_2/selector/greedy_prefix_survival": 0.5,
            "position_2/hard_label/unary_top1_accuracy": 0.8,
            "position_2/hard_label/unary_top16_oracle_prefix_survival": 1.0,
            "position_3/selector/greedy_prefix_acceptance": 0.0,
            "position_10/selector/greedy_prefix_acceptance": 0.25,
            "position_2/objective/loss_weight_share": 0.4,
            "dflash/teacher/unary_distribution_overlap": 0.6,
        },
        "ratio_totals": {
            "ce_loss": [20, 10],
            "position_2/selector/greedy_prefix_acceptance": [2, 3],
            "position_2/selector/greedy_prefix_survival": [2, 4],
            "position_3/selector/greedy_prefix_acceptance": [0, 0],
        },
        "sum_totals": {"position_2/selector/greedy_reached_count": 3},
    }


def test_group_names_preserve_distinct_populations_and_raw_report(report):
    original = copy.deepcopy(report)
    metrics = checkpoint_metrics(report)
    assert metrics["eval/summary/sample_mean_loss"] == 2.5
    assert metrics["eval/summary/ce_loss"] == 2
    assert metrics["eval/summary/selector_greedy_accepted_length"] == 2.25
    assert metrics["eval/oracle/top16_accepted_length"] == 3.5
    assert metrics["eval/position_2/selector_prefix_acceptance"] == 2 / 3
    assert metrics["eval/position_2/selector_prefix_survival"] == 0.5
    assert metrics["eval/position_2/unary_top1_accuracy"] == 0.8
    assert metrics["eval/position_2/oracle_top16_prefix_survival"] == 1
    assert not any("position_3/" in key for key in metrics)
    assert not any("diagnostics" in key for key in metrics)
    detailed = checkpoint_metrics(report, include_diagnostics=True)
    assert detailed["eval/position_2/diagnostics/objective/loss_weight_share"] == 0.4
    assert detailed["eval/position_2/counts/selector/greedy_reached_count"] == 3
    assert detailed["eval/diagnostics/dflash/teacher/unary_distribution_overlap"] == 0.6
    assert original == report


def test_markdown_puts_summary_first_and_sorts_positions_numerically(report):
    markdown = render_checkpoint_report(report)
    assert markdown.index("Composite loss") < markdown.index("| Position")
    assert markdown.index("| 2 |") < markdown.index("| 10 |")
    assert "66.67% | 50.00%" in markdown
    assert "| 3 | — | — | — | — | — |" in markdown
    assert "candidate-coverage bound" in markdown
    assert "total evaluated blocks (including censored tails)" in markdown


@pytest.mark.parametrize("top_k", [4, 16, 32])
def test_oracle_names_follow_actual_candidate_width(report, top_k):
    report["mean"] = {
        name.replace("top16", f"top{top_k}"): value
        for name, value in report["mean"].items()
    }
    metrics = checkpoint_metrics(report)
    assert metrics[f"eval/oracle/top{top_k}_accepted_length"] == 3.5
    assert metrics[f"eval/position_2/oracle_top{top_k}_prefix_survival"] == 1


@pytest.mark.parametrize(
    "bad",
    [
        "failed",
        "empty",
        "nan",
        "inconsistent",
        "negative_denominator",
        "nonzero_over_zero",
    ],
)
def test_reject_invalid_saved_reports(report, bad):
    if bad == "failed":
        report["success"] = False
    elif bad == "empty":
        report["examples"] = 0
    elif bad == "nan":
        report["mean"]["loss"] = float("nan")
    elif bad == "inconsistent":
        report["mean"]["ce_loss"] = 999
    elif bad == "negative_denominator":
        report["ratio_totals"]["ce_loss"] = [20, -10]
    else:
        report["ratio_totals"]["ce_loss"] = [20, 0]
    with pytest.raises(ValueError):
        checkpoint_metrics(report)


def test_legacy_report_without_counts_and_plain_dflash_remain_readable(report):
    del report["sum_totals"]
    report["mean"] = {
        "loss": 1.5,
        "dflash/hard_label/unary_greedy_accepted_length": 2.5,
    }
    report["ratio_totals"] = {}
    metrics = checkpoint_metrics(report)
    assert metrics["eval/summary/unary_greedy_accepted_length"] == 2.5
    assert "selector" not in json.dumps(metrics)
    assert "Unary greedy" in render_checkpoint_report(report)


def test_cli_reformats_existing_json_and_refuses_output_collisions(report, tmp_path):
    source = tmp_path / "raw.json"
    source.write_text(json.dumps(report))
    markdown, metrics = tmp_path / "report.md", tmp_path / "metrics.json"
    args = [
        "eval-report",
        "--input",
        str(source),
        "--output",
        str(markdown),
        "--metrics-output",
        str(metrics),
    ]
    assert main(args) == 0
    assert json.loads(metrics.read_text()) == checkpoint_metrics(report)
    assert markdown.read_text() == render_checkpoint_report(report)
    assert json.loads(source.read_text()) == report
    assert main(args) == 2
    assert (
        main(
            [
                "eval-report",
                "--input",
                str(source),
                "--output",
                str(tmp_path / "new"),
                "--metrics-output",
                str(tmp_path / "new"),
            ]
        )
        == 2
    )
