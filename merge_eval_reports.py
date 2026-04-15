"""
Merge two eval_report.json files (same dataset order) into eval_report.dual.json for report_viewer.html.

Usage:
  python merge_eval_reports.py --a eval_report_glm.json --b eval_report_oss.json -o eval_report.dual.json
  python merge_eval_reports.py --a a.json --b b.json --label-a "GLM 4.7" --label-b "OSS"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


RUN_SUMMARY_KEYS = (
    "status",
    "fail_reason",
    "failed_example_index",
    "failed_example_indices",
    "average_score",
    "evaluated_count",
    "total_count",
    "target_model",
    "target_provider",
    "avg_target_latency_s",
    "avg_target_latency_ms",
    "sum_target_tokens",
    "avg_target_tokens_per_example",
    "examples_with_latency",
    "examples_with_token_usage",
    "max_examples",
    "dataset_rows_in_file",
    "dataset_rows_evaluated",
)

PER_MODEL_EXAMPLE_KEYS = (
    "target_output",
    "judge_score",
    "judge_reason",
    "judge_compliant",
    "is_sensitive",
    "target_latency_ms",
    "target_prompt_tokens",
    "target_completion_tokens",
    "target_total_tokens",
)


def load_report(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: root must be an object.")
    return data


def extract_run_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in RUN_SUMMARY_KEYS:
        if k in report:
            out[k] = report[k]
    return out


def extract_per_model_example(ex: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in PER_MODEL_EXAMPLE_KEYS:
        if k in ex:
            out[k] = ex[k]
    return out


def merge_reports(
    report_a: Dict[str, Any],
    report_b: Dict[str, Any],
    label_a: Optional[str],
    label_b: Optional[str],
) -> Dict[str, Any]:
    for key in ("judge_model", "judge_provider"):
        if report_a.get(key) != report_b.get(key):
            raise ValueError(
                f"Reports differ on {key!r}: {report_a.get(key)!r} vs {report_b.get(key)!r}. "
                "Use the same judge for both runs."
            )

    if report_a.get("total_count") != report_b.get("total_count"):
        raise ValueError(
            f"Reports differ on 'total_count': {report_a.get('total_count')!r} vs {report_b.get('total_count')!r}."
        )

    ev_a = report_a.get("evaluated_count")
    ev_b = report_b.get("evaluated_count")
    if ev_a != ev_b:
        print(
            f"Warning: evaluated_count differs ({ev_a!r} vs {ev_b!r}). "
            "Usually one run stopped early on judge (e.g. fail on a sensitive prompt). "
            "Merge continues if example rows align.",
            file=sys.stderr,
        )

    ex_a = report_a.get("examples") or []
    ex_b = report_b.get("examples") or []
    if not isinstance(ex_a, list) or not isinstance(ex_b, list):
        raise ValueError("Both reports must have an 'examples' array.")
    if len(ex_a) != len(ex_b):
        raise ValueError(f"Example count mismatch: {len(ex_a)} vs {len(ex_b)}.")

    merged_examples: List[Dict[str, Any]] = []
    for i, (a, b) in enumerate(zip(ex_a, ex_b)):
        if not isinstance(a, dict) or not isinstance(b, dict):
            raise ValueError(f"Example {i}: expected objects.")
        for field in ("prompt", "expected_answer"):
            if a.get(field) != b.get(field):
                raise ValueError(
                    f"Example {i}: {field!r} mismatch between reports "
                    f"(same dataset order is required)."
                )
        merged_examples.append(
            {
                "index": a.get("index", i),
                "prompt": a.get("prompt"),
                "expected_answer": a.get("expected_answer"),
                "category": a.get("category"),
                "per_model": [extract_per_model_example(a), extract_per_model_example(b)],
            }
        )

    la = label_a or str(report_a.get("target_model") or "Model A")
    lb = label_b or str(report_b.get("target_model") or "Model B")

    return {
        "format": "dual",
        "judge_model": report_a.get("judge_model"),
        "judge_provider": report_a.get("judge_provider"),
        "run_labels": [la, lb],
        "runs": [extract_run_summary(report_a), extract_run_summary(report_b)],
        "examples": merged_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge two eval_report.json files for dual report_viewer.")
    parser.add_argument("--a", type=Path, required=True, help="First report JSON (e.g. GLM run).")
    parser.add_argument("--b", type=Path, required=True, help="Second report JSON (e.g. OSS run).")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("eval_report.dual.json"),
        help="Output path (default: eval_report.dual.json).",
    )
    parser.add_argument("--label-a", default=None, help="Display label for run A (default: target_model from A).")
    parser.add_argument("--label-b", default=None, help="Display label for run B (default: target_model from B).")
    args = parser.parse_args()

    ra = load_report(args.a)
    rb = load_report(args.b)
    merged = merge_reports(ra, rb, args.label_a, args.label_b)

    args.out.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(merged['examples'])} merged examples to {args.out.resolve()}")


if __name__ == "__main__":
    main()
