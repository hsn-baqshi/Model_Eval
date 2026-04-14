"""
Infer a ``category`` field for each ``{prompt, answer}`` object in a dataset JSON array.

Usage:
  python assign_dataset_categories.py 20260416_evaluation_dataset.json
  python assign_dataset_categories.py INPUT.json --out OUTPUT.json

Categories are lowercase slugs aligned with ``llm_eval_framework`` sensitivity heuristics
(``security`` for policy/refusal-style items; ``privacy`` only when clearly PII-focused).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def infer_category(prompt: str, answer: str) -> str:
    p = (prompt or "").strip()
    a = (answer or "").strip()
    pl = p.lower()
    al = a.lower()

    # Harm / policy evaluation pairs in this dataset use a fixed refusal answer.
    if al == "it is not allowed." or al.startswith("it is not allowed"):
        return "security"

    # Numbered template prefixes (synthetic eval slices)
    if re.match(r"^Scenario\s*#", p, re.I):
        return "philosophy"
    if re.match(r"^Task\s*#", p, re.I):
        return "productivity"
    if re.match(r"^Concept\s*#", p, re.I):
        return "science"
    if re.match(r"^Fact check\s*#", p, re.I):
        return "science"

    # Programming tracks (language tutorials / syntax)
    if re.search(r"\bpython\b", pl) or re.search(r"\bjavascript\b", pl):
        return "programming"
    if re.search(r"\bjava\b", pl) and "javascript" not in pl:
        return "programming"
    if "c++" in pl or re.search(r"\bc#\b", pl):
        return "programming"

    philosophy_kw = (
        "utilitarian",
        "existentialism",
        "objectivity",
        "free will",
        "determinism",
        "trolley problem",
        "does the end justify",
        "meaning of life",
    )
    if any(k in pl for k in philosophy_kw):
        return "philosophy"

    business_kw = (
        "business loan",
        "small business loan",
        "marketing strategy",
        "eco-friendly water bottle",
        "adam smith",
        "john maynard keynes",
        "economic theories",
        "filing for",
    )
    if any(k in pl for k in business_kw):
        return "business"

    if "yellow wallpaper" in pl or ("themes" in pl and "isolation" in pl):
        return "literature"

    if "500-word essay" in pl or "industrial revolution" in pl:
        return "academic_writing"

    lifestyle_kw = (
        "recipe",
        "wine stain",
        "carpet",
        "vegan chocolate",
        "morning routine",
        "to-do list",
    )
    if any(k in pl for k in lifestyle_kw):
        return "lifestyle"

    if "summarize the plot" in pl or "in three sentences" in pl:
        return "media"

    if "formal email" in pl or "refund" in pl:
        return "professional_writing"

    science_kw = (
        "photosynthesis",
        "boiling point",
        "largest planet",
        "solar system",
        "butterfly effect",
        "speed of light",
    )
    if any(k in pl for k in science_kw):
        return "science"

    if "compare and contrast" in pl and ("python" in pl or "java" in pl):
        return "programming"

    # Trivia / general knowledge
    if any(
        x in pl
        for x in (
            "capital of",
            "who wrote",
            "romeo and juliet",
        )
    ):
        return "general"

    return "general"


def annotate_dataset(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in items:
        if not isinstance(row, dict):
            raise ValueError("Each dataset row must be an object.")
        prompt = str(row.get("prompt", ""))
        answer = str(row.get("answer", ""))
        cat = infer_category(prompt, answer)
        new_row = dict(row)
        new_row["category"] = cat
        out.append(new_row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Add inferred category to each dataset item.")
    parser.add_argument("input", type=Path, help="Input JSON array path.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: overwrite input).",
    )
    args = parser.parse_args()
    inp = args.input
    out_path = args.out or inp

    raw = json.loads(inp.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Dataset must be a JSON array of objects.")
    annotated = annotate_dataset(raw)
    out_path.write_text(json.dumps(annotated, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(annotated)} rows to {out_path.resolve()}")


if __name__ == "__main__":
    main()
