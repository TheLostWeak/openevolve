"""
Aggregate per-iteration prompt/response JSON files into a single JSON and Markdown file.

Usage:
    python scripts/aggregate_prompts.py \ 
        --prompts-dir examples/cap_set_example/openevolve_output/db/prompts \ 
        --out-json examples/cap_set_example/openevolve_output/db/prompts_aggregated.json \ 
        --out-md examples/cap_set_example/openevolve_output/db/prompts_aggregated.md

If no arguments are provided, defaults target the `examples/cap_set_example` output.
"""
import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Any


def load_prompt_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        try:
            # Many files are JSON; try to parse JSON first
            return json.load(fh)
        except Exception:
            # Fall back to raw text
            fh.seek(0)
            return {"raw": fh.read()}


def aggregate(prompts_dir: str) -> List[Dict[str, Any]]:
    entries = []
    if not os.path.isdir(prompts_dir):
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    files = [f for f in os.listdir(prompts_dir) if f.lower().endswith(".json")]
    # Sort files by name (iterX_... will be roughly chronological), fallback to mtime
    files.sort()

    for fname in files:
        p = os.path.join(prompts_dir, fname)
        try:
            data = load_prompt_file(p)
            entry = {"file": fname, "path": p, "content": data}
            # normalize saved_at if present
            saved = None
            if isinstance(data, dict) and "saved_at" in data:
                try:
                    saved = float(data.get("saved_at"))
                except Exception:
                    saved = None
            elif os.path.exists(p):
                saved = os.path.getmtime(p)

            if saved:
                entry["saved_at"] = saved
                entry["saved_at_iso"] = datetime.fromtimestamp(saved).isoformat()

            entries.append(entry)
        except Exception as e:
            entries.append({"file": fname, "path": p, "error": str(e)})

    return entries


def write_outputs(entries: List[Dict[str, Any]], out_json: str, out_md: str) -> None:
    # Write JSON
    with open(out_json, "w", encoding="utf-8") as jfh:
        json.dump(entries, jfh, ensure_ascii=False, indent=2)

    # Write human-readable markdown
    with open(out_md, "w", encoding="utf-8") as mfh:
        mfh.write("# Aggregated Prompts and Responses\n\n")
        mfh.write(f"Generated: {datetime.now().isoformat()}\n\n")

        for e in entries:
            mfh.write(f"## File: {e.get('file')}\n\n")
            if "saved_at_iso" in e:
                mfh.write(f"- saved_at: {e['saved_at_iso']}\n")
            if "error" in e:
                mfh.write(f"- error: {e['error']}\n\n")
                continue

            content = e.get("content") or {}
            # Write prompt.system and prompt.user if present
            prompt = content.get("prompt") if isinstance(content, dict) else None
            if prompt and isinstance(prompt, dict):
                mfh.write("### System Prompt\n\n")
                mfh.write("```")
                mfh.write(str(prompt.get("system", "")))
                mfh.write("```\n\n")

                mfh.write("### User Prompt\n\n")
                mfh.write("```")
                mfh.write(str(prompt.get("user", "")))
                mfh.write("```\n\n")

            # Responses
            responses = content.get("responses") if isinstance(content, dict) else None
            if responses:
                for i, r in enumerate(responses):
                    mfh.write(f"### Response {i+1}\n\n")
                    mfh.write("```")
                    # responses may be long; write as-is
                    mfh.write(str(r))
                    mfh.write("```\n\n")
            else:
                # If the file has a top-level 'responses' absent, try 'raw' or 'llm_response'
                if isinstance(content, dict) and "llm_response" in content:
                    mfh.write("### LLM Response\n\n```")
                    mfh.write(str(content.get("llm_response")))
                    mfh.write("```\n\n")
                elif isinstance(content, dict) and "raw" in content:
                    mfh.write("```")
                    mfh.write(str(content.get("raw")))
                    mfh.write("```\n\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-iteration prompts/responses")
    parser.add_argument("--prompts-dir", default="examples/cap_set_example/openevolve_output/db/prompts")
    parser.add_argument("--out-json", default="examples/cap_set_example/openevolve_output/db/prompts_aggregated.json")
    parser.add_argument("--out-md", default="examples/cap_set_example/openevolve_output/db/prompts_aggregated.md")
    args = parser.parse_args()

    entries = aggregate(args.prompts_dir)
    write_outputs(entries, args.out_json, args.out_md)
    print(f"Wrote {len(entries)} entries to {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
