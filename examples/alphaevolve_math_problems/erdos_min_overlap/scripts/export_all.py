#!/usr/bin/env python3
"""
Export per-iteration artifacts and a compact CSV summary.

Usage:
  python export_all.py --output <openevolve_output_dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_load_error": str(exc)}


def _safe_write(path: Path, content: Optional[str], placeholder: str = "[MISSING]") -> None:
    path.write_text((content if content is not None else placeholder) or "", encoding="utf-8")


def _latest_by_timestamp(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not items:
        return None
    return max(items, key=lambda x: x.get("timestamp", 0))


def _find_prompt_for_program(prompts_dir: Path, program_id: str) -> Optional[Path]:
    candidates = sorted(prompts_dir.glob(f"{program_id}_*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _extract_prompt_data(prompt_path: Optional[Path]) -> Dict[str, Optional[str]]:
    if not prompt_path:
        return {
            "system": None,
            "user": None,
            "response": None,
            "meta_prompt": None,
            "meta_prompt_id": None,
            "meta_prompt_text": None,
        }
    data = _load_json(prompt_path)
    prompt = data.get("prompt", {})
    responses = prompt.get("responses") or data.get("responses") or []
    first_response = responses[0] if responses else None
    return {
        "system": prompt.get("system"),
        "user": prompt.get("user"),
        "response": first_response,
        "meta_prompt": prompt.get("meta_prompt"),
        "meta_prompt_id": prompt.get("meta_prompt_id"),
        "meta_prompt_text": prompt.get("meta_prompt_text"),
    }


def _extract_meta_prompt_from_user(user_text: Optional[str]) -> Optional[str]:
    if not user_text:
        return None
    marker = "## Meta Prompt (evolving)"
    if marker not in user_text:
        return None
    # Capture block until next blank line.
    pattern = re.compile(r"## Meta Prompt \\(evolving\\)\\n(.*?)(\\n\\n|$)", re.S)
    match = pattern.search(user_text)
    if not match:
        return None
    return match.group(1).strip()


def _write_iteration_artifacts(
    export_dir: Path, iteration: int, program: Dict[str, Any], prompt_data: Dict[str, Optional[str]]
) -> None:
    iter_dir = export_dir / f"iter{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    _safe_write(iter_dir / "prompt_system.txt", prompt_data.get("system"))
    _safe_write(iter_dir / "prompt_user.txt", prompt_data.get("user"))
    _safe_write(iter_dir / "response.txt", prompt_data.get("response"))
    _safe_write(iter_dir / "meta_prompt.txt", prompt_data.get("meta_prompt"))
    _safe_write(iter_dir / "meta_prompt_id.txt", prompt_data.get("meta_prompt_id"))
    _safe_write(iter_dir / "meta_prompt_text.txt", prompt_data.get("meta_prompt_text"))

    code = program.get("code") or ""
    _safe_write(iter_dir / "final_code.py", code, placeholder="")

    metrics = program.get("metrics") or {}
    if metrics:
        (iter_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _write_summary_csv(output_path: Path, rows: List[Dict[str, Any]]) -> None:
    def _format_field(value: Any) -> Any:
        # Increase numeric precision for CSV output while preserving non-numeric fields.
        if isinstance(value, (int, float)):
            return f"{value:.12f}"
        return value if value is not None else ""

    fieldnames = ["iteration", "combined_score", "c5_bound", "n_points", "eval_time", "error"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["iteration"]):
            writer.writerow({k: _format_field(row.get(k)) for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", required=True, help="Path to openevolve_output")
    parser.add_argument(
        "--export-dir",
        "-e",
        default=None,
        help="Directory for per-iteration exports (defaults to sibling exports/)",
    )
    args = parser.parse_args()

    out_base = Path(args.output).resolve()
    programs_dir = out_base / "db" / "programs"
    prompts_dir = out_base / "db" / "prompts"

    export_dir = Path(args.export_dir).resolve() if args.export_dir else out_base.parent / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    programs_by_iter: Dict[int, List[Dict[str, Any]]] = {}
    for path in programs_dir.glob("*.json"):
        program = _load_json(path)
        iteration = program.get("iteration_found")
        if iteration is None:
            continue
        try:
            iteration = int(iteration)
        except Exception:
            continue
        programs_by_iter.setdefault(iteration, []).append(program)

    summary_rows: List[Dict[str, Any]] = []
    for iteration, candidates in programs_by_iter.items():
        program = _latest_by_timestamp(candidates)
        if not program:
            continue
        program_id = program.get("id", "")
        prompt_path = _find_prompt_for_program(prompts_dir, str(program_id))
        prompt_data = _extract_prompt_data(prompt_path)
        if not prompt_data.get("meta_prompt"):
            extracted = _extract_meta_prompt_from_user(prompt_data.get("user"))
            if extracted:
                prompt_data["meta_prompt"] = extracted
        if not prompt_data.get("meta_prompt_text"):
            prompt_data["meta_prompt_text"] = prompt_data.get("meta_prompt")
        _write_iteration_artifacts(export_dir, iteration, program, prompt_data)

        metrics = program.get("metrics") or {}
        summary_rows.append(
            {
                "iteration": iteration,
                "combined_score": metrics.get("combined_score"),
                "c5_bound": metrics.get("c5_bound"),
                "n_points": metrics.get("n_points"),
                "eval_time": metrics.get("eval_time"),
                "error": metrics.get("error"),
            }
        )

    summary_path = out_base / "iteration_summary.csv"
    _write_summary_csv(summary_path, summary_rows)
    export_summary_path = export_dir / "iteration_summary.csv"
    export_summary_path.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Wrote summary {summary_path} with {len(summary_rows)} iterations")
    print(f"Wrote exports into {export_dir}")


if __name__ == "__main__":
    main()
