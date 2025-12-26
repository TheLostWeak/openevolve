#!/usr/bin/env python3
"""Unified export script for cap_set_example

Usage:
  python export_all.py --output <openevolve_output_dir>

Creates per-iteration directories under <output>/iteration_exports/iter{n} containing:
  - prompt.txt        (System / User separated, unescaped newlines)
  - response.json     (if response is JSON)
  - response.txt      (plain text response, unescaped)
  - final_code.py/.txt (extracted generated code, unescaped)

Also writes iteration_summary.csv with references.
"""
from __future__ import annotations

import argparse
import glob
import json
import ast
import os
import shutil
import re
import csv
from typing import Any, Dict, List, Optional, Tuple


def find_prompt_files(prompts_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(prompts_dir, "*.json")))


def safe_load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {'_load_error': str(e)}


def extract_iter(fname: str) -> Optional[int]:
    b = os.path.basename(fname)
    m = re.search(r"iter(\d+)_", b)
    if m:
        return int(m.group(1))
    m2 = re.search(r"iter(\d+)", b)
    if m2:
        return int(m2.group(1))
    return None


def pick_prompt_text(entry: Dict[str, Any]) -> str:
    # prefer common fields
    for key in ("prompt", "system_prompt", "system", "user", "user_prompt", "full_prompt"):
        if key in entry and entry[key]:
            if isinstance(entry[key], (list, dict)):
                return json.dumps(entry[key], ensure_ascii=False, indent=2)
            return str(entry[key])
    parts = []
    if 'system' in entry:
        parts.append(str(entry.get('system')))
    if 'user' in entry:
        parts.append(str(entry.get('user')))
    if parts:
        return "\n\n".join(parts)
    return json.dumps(entry, ensure_ascii=False, indent=2)


def pick_response_text(entry: Dict[str, Any]) -> str:
    for key in ("response", "llm_response", "raw_response", "message", "raw_llm", "reply", "responses"):
        if key in entry and entry[key] is not None:
            val = entry[key]
            if isinstance(val, (dict, list)):
                # try to get content fields from model output
                if isinstance(val, dict):
                    cont = val.get('content') or val.get('text') or val.get('reasoning')
                    if cont:
                        return str(cont)
                return json.dumps(val, ensure_ascii=False, indent=2)
            return str(val)
    if 'provider_repr' in entry:
        return str(entry.get('provider_repr'))
    # fallback: dump JSON
    return json.dumps(entry, ensure_ascii=False, indent=2)


def explain_response(entry: Dict[str, Any], response_text: Optional[str]) -> str:
    """Return a human-friendly explanation / rendering of the response as plain text."""
    # Prefer structured `responses` or similar
    try:
        if isinstance(entry, dict):
            for key in ('responses', 'response', 'raw_response', 'llm_response', 'raw_llm', 'reply'):
                if key in entry and entry[key] is not None:
                    val = entry[key]
                    if isinstance(val, (dict, list)):
                        # Build a readable summary
                        out_lines: List[str] = []
                        out_lines.append(f"Response field: {key}")
                        if isinstance(val, dict):
                            # try to pull model/content-like fields
                            model = val.get('model') or val.get('provider') or val.get('engine')
                            if model:
                                out_lines.append(f"Model: {model}")
                            # content may be nested
                            if 'content' in val:
                                content = val.get('content')
                                out_lines.append('\n-- Content --')
                                out_lines.append(str(content))
                            else:
                                out_lines.append('\n-- JSON --')
                                out_lines.append(json.dumps(val, ensure_ascii=False, indent=2))
                        else:
                            # list: enumerate
                            out_lines.append('\n-- Items --')
                            for i, item in enumerate(val):
                                out_lines.append(f"[{i}]")
                                if isinstance(item, (dict, list)):
                                    # try to surface common text fields
                                    if isinstance(item, dict):
                                        cont = item.get('content') or item.get('text') or item.get('message')
                                        if cont:
                                            out_lines.append(str(cont))
                                        else:
                                            out_lines.append(json.dumps(item, ensure_ascii=False, indent=2))
                                    else:
                                        out_lines.append(json.dumps(item, ensure_ascii=False, indent=2))
                                else:
                                    out_lines.append(str(item))

                        return '\n'.join(out_lines)
                    else:
                        return str(val)

        # If we reach here, try to parse response_text as JSON and explain
        if response_text:
            try:
                parsed = json.loads(response_text)
                return 'Parsed response JSON:\n' + json.dumps(parsed, ensure_ascii=False, indent=2)
            except Exception:
                # plain text path
                return unescape_text(response_text) or ''
    except Exception:
        pass
    # fallback: raw
    return unescape_text(response_text) or ''


def unescape_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return s
    if not isinstance(s, str):
        return s
    try:
        # Common escaped sequences first
        s2 = s.replace('\\r\\n', '\r\n').replace('\\n', '\n').replace('\\t', '\t')
        # Unescape escaped quotes and slashes
        s2 = s2.replace('\\"', '"').replace("\\'", "'")
        # Collapse doubled escapes (\\ -> \\)
        s2 = s2.replace('\\\\', '\\')
        # Final pass: decode unicode escapes (safe)
        try:
            s3 = bytes(s2, 'utf-8').decode('unicode_escape')
        except Exception:
            s3 = s2
        return s3
    except Exception:
        return s


def extract_code_from_entry(entry: Dict[str, Any]) -> Optional[str]:
    # 1) program DB fields
    for k in ('final_code', 'source', 'code', 'source_code', 'program', 'generated_code', 'module_text', 'module_source'):
        if k in entry and entry[k]:
            v = entry[k]
            if isinstance(v, (list, dict)):
                return json.dumps(v, ensure_ascii=False, indent=2)
            return str(v)
    # 2) responses field - look for fenced code
    resp = pick_response_text(entry)
    if isinstance(resp, str):
        m = re.search(r"```(?:python\n)?(.*?)```", resp, flags=re.S)
        if m:
            return m.group(1).strip()
        # heuristic: if looks like code
        if ('def ' in resp or 'class ' in resp or 'import ' in resp) and resp.count('\n') > 1:
            return resp
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', '-o', required=True, help='Path to example openevolve_output (used to find db/prompts and db/programs)')
    p.add_argument('--export-dir', '-e', dest='export_dir', default=None,
                   help='Directory to write per-iteration exports (defaults to <output>/iteration_exports)')
    args = p.parse_args()

    out_base = os.path.abspath(args.output)
    prompts_dir = os.path.join(out_base, 'db', 'prompts')
    programs_dir = os.path.join(out_base, 'db', 'programs')
    # NOTE: no longer create `iteration_prompts` or `iteration_responses` directories.
    # We produce per-iteration exports under `iteration_exports` only.
    # Determine exports directory: allow overriding so exports needn't live
    # under the `openevolve_output` tree.
    if args.export_dir:
        out_exports = os.path.abspath(args.export_dir)
    else:
        # default exports directory: sibling `exports` next to the provided
        # `openevolve_output` directory. This keeps exported artifacts out of
        # the openevolve output folder by default.
        out_exports = os.path.join(os.path.dirname(out_base), 'exports')
    os.makedirs(out_exports, exist_ok=True)

    prompt_files = []
    if os.path.isdir(prompts_dir):
        prompt_files = find_prompt_files(prompts_dir)

    per_iter: Dict[int, List[Tuple[str, Dict[str, Any]]]] = {}

    for pf in prompt_files:
        entry = safe_load_json(pf)
        it = extract_iter(pf)
        if it is None:
            # try to read iteration from json fields
            for fk in ('iteration', 'iter', 'iteration_num'):
                if fk in entry:
                    try:
                        it = int(entry[fk])
                    except Exception:
                        it = None
        if it is None:
            continue
        per_iter.setdefault(it, []).append((pf, entry))

    # NOTE: we intentionally no longer consume legacy `iteration_prompts/iterN.txt`
    # files. Per-iteration exports are written under `iteration_exports` and
    # iteration discovery should come from `db/prompts/*.json`.

    # programs lookup
    prog_lookup = {}
    if os.path.isdir(programs_dir):
        for pf in glob.glob(os.path.join(programs_dir, '*.json')):
            j = safe_load_json(pf)
            pid = j.get('id') or j.get('program_id') or j.get('uuid')
            if pid:
                prog_lookup[str(pid)] = j

    rows = []

    for it in sorted(per_iter.keys()):
        items = per_iter[it]
        # Choose the most recently modified file for this iteration (mtime).
        # Items are tuples (filepath, entry). Use os.path.getmtime when possible.
        def _mtime_key(item):
            try:
                return os.path.getmtime(item[0])
            except Exception:
                return 0

        items = sorted(items, key=_mtime_key)
        pf, entry = items[-1]

        program_id = entry.get('program_id') or entry.get('id') or entry.get('uuid')
        # try to infer program id from prompt filename if missing (filenames often contain the uuid)
        if not program_id and pf:
            m = re.search(r"([0-9a-fA-F\-]{36})", os.path.basename(pf))
            if m:
                maybe = m.group(1)
                if maybe in prog_lookup:
                    program_id = maybe

        # Discover full metrics dict for this iteration. We prefer structured
        # metrics stored on the prompt entry under common keys, and fall back
        # to the program DB if not present. This `full_metrics` is reused both
        # for writing `metrics.json` and for populating CSV fields like
        # `combined_score` to keep behavior consistent.
        full_metrics = None
        if isinstance(entry, dict):
            # look for likely metric containers on the entry
            for fk in ('metrics', 'evaluation', 'result', 'scores'):
                if fk in entry and isinstance(entry[fk], dict):
                    full_metrics = entry[fk]
                    break

        # fall back to program DB if needed
        if full_metrics is None and program_id and str(program_id) in prog_lookup:
            try:
                pj = prog_lookup[str(program_id)]
                for fk in ('metrics', 'evaluation'):
                    if fk in pj and isinstance(pj[fk], dict):
                        full_metrics = pj[fk]
                        break
            except Exception:
                full_metrics = None

        combined = None
        size = None
        if isinstance(full_metrics, dict):
            combined = full_metrics.get('combined_score')
            size = full_metrics.get('size')

        # pick prompt and response
        prompt_text = pick_prompt_text(entry)
        response_text = pick_response_text(entry)

        iter_dir = os.path.join(out_exports, f'iter{it}')
        os.makedirs(iter_dir, exist_ok=True)

        # write prompt with unescaped newlines
        # Determine system vs user prompt robustly
        def resolve_system_user(entry_obj: Dict[str, Any], prompt_blob: str) -> Tuple[str, str]:
            # explicit fields preferred
            sys_p = None
            user_p = None
            for k in ('system', 'system_prompt'):
                if k in entry_obj and entry_obj[k]:
                    sys_p = entry_obj[k]
                    break
            for k in ('user', 'user_prompt'):
                if k in entry_obj and entry_obj[k]:
                    user_p = entry_obj[k]
                    break

            # if explicit values are dict/list, stringify
            if isinstance(sys_p, (dict, list)):
                sys_p = json.dumps(sys_p, ensure_ascii=False, indent=2)
            if isinstance(user_p, (dict, list)):
                user_p = json.dumps(user_p, ensure_ascii=False, indent=2)

            # if both found, return
            if sys_p or user_p:
                return (sys_p or '', user_p or '')

            # try to parse prompt_blob if it's JSON
            # if prompt_blob is already a dict (some prompt files store a dict under 'prompt')
            if isinstance(prompt_blob, dict):
                sys_c = prompt_blob.get('system') or prompt_blob.get('system_prompt')
                user_c = prompt_blob.get('user') or prompt_blob.get('user_prompt') or prompt_blob.get('prompt')
                if sys_c or user_c:
                    return (
                        json.dumps(sys_c, ensure_ascii=False, indent=2)
                        if isinstance(sys_c, (dict, list))
                        else (sys_c or ''),
                        json.dumps(user_c, ensure_ascii=False, indent=2)
                        if isinstance(user_c, (dict, list))
                        else (user_c or ''),
                    )

            # try to parse prompt_blob if it's JSON-like string
            if isinstance(prompt_blob, str) and prompt_blob.strip().startswith('{'):
                parsed = None
                try:
                    parsed = json.loads(prompt_blob)
                except Exception:
                    # try to parse Python-style dict literal (single quotes)
                    try:
                        parsed = ast.literal_eval(prompt_blob)
                    except Exception:
                        parsed = None

                if isinstance(parsed, dict):
                    sys_c = parsed.get('system') or parsed.get('system_prompt')
                    user_c = parsed.get('user') or parsed.get('user_prompt') or parsed.get('prompt')
                    if sys_c or user_c:
                        return (
                            json.dumps(sys_c, ensure_ascii=False, indent=2)
                            if isinstance(sys_c, (dict, list))
                            else (sys_c or ''),
                            json.dumps(user_c, ensure_ascii=False, indent=2)
                            if isinstance(user_c, (dict, list))
                            else (user_c or ''),
                        )

            # look for explicit markers in text
            if isinstance(prompt_blob, str):
                m = re.search(r"=== *System Prompt *=+\n(.*?)\n=== *User Prompt *=+\n(.*)", prompt_blob, flags=re.S)
                if m:
                    return (m.group(1).strip(), m.group(2).strip())

                # split on double newline as heuristic
                parts = re.split(r"\n\s*\n", prompt_blob.strip(), maxsplit=1)
                if len(parts) == 2:
                    # assume first is system, second is user
                    return (parts[0].strip(), parts[1].strip())

            # fallback: no system, entire blob is user
            return ('', prompt_blob if isinstance(prompt_blob, str) else str(prompt_blob))

        try:
            raw_prompt_blob = entry.get('prompt') if 'prompt' in entry else prompt_text
            sys_prompt, user_prompt = resolve_system_user(entry, raw_prompt_blob)
        except Exception:
            sys_prompt = ''
            user_prompt = prompt_text

        sys_prompt = unescape_text(sys_prompt).strip()
        user_prompt = unescape_text(user_prompt).strip()

        with open(os.path.join(iter_dir, 'prompt.txt'), 'w', encoding='utf-8') as pfh:
            pfh.write('=== System Prompt ===\n\n')
            pfh.write(sys_prompt + '\n\n')
            pfh.write('=== User Prompt ===\n\n')
            pfh.write(user_prompt + '\n')

        # write response as human-readable text (explanation), avoid raw JSON file output
        resp_txt_path = os.path.join(iter_dir, 'response.txt')
        try:
            explained = explain_response(entry if isinstance(entry, dict) else {}, response_text)
            with open(resp_txt_path, 'w', encoding='utf-8') as rt:
                rt.write((explained or '').rstrip() + '\n')
        except Exception:
            # best-effort fallback
            with open(resp_txt_path, 'w', encoding='utf-8') as rt:
                rt.write((unescape_text(response_text) or '').rstrip() + '\n')

            # remove any JSON response leftover to avoid duplication
            try:
                resp_json_path = os.path.join(iter_dir, 'response.json')
                if os.path.exists(resp_json_path):
                    os.remove(resp_json_path)
            except Exception:
                pass

        

        # write full metrics JSON if we discovered a metrics dict earlier
        metrics_path = None
        try:
            if isinstance(full_metrics, dict) and full_metrics:
                metrics_path = os.path.join(iter_dir, 'metrics.json')
                try:
                    with open(metrics_path, 'w', encoding='utf-8') as mf:
                        json.dump(full_metrics, mf, ensure_ascii=False, indent=2)
                except Exception:
                    metrics_path = None
        except Exception:
            metrics_path = None

        # remove any old result.txt files to avoid confusion (we rely on metrics.json)
        try:
            old_result = os.path.join(iter_dir, 'result.txt')
            if os.path.exists(old_result):
                os.remove(old_result)
        except Exception:
            pass

        # extract final code
        final_code = None
        # prefer program DB
        source_blob = None
        if program_id and str(program_id) in prog_lookup:
            pj = prog_lookup[str(program_id)]
            final_code = extract_code_from_entry(pj)
            # try to get a full textual blob from program DB for diff/context
            # prefer fields that look like source text
            for src_k in ('source', 'source_code', 'module_source', 'module_text', 'program'):
                if src_k in pj and pj[src_k]:
                    source_blob = pj[src_k]
                    break
            if source_blob is None:
                source_blob = json.dumps(pj, ensure_ascii=False, indent=2)
        if not final_code:
            final_code = extract_code_from_entry(entry)
            # use entry-based full blob for diff/context
            if 'program' in entry and entry['program']:
                source_blob = entry['program']
            else:
                # prefer response_text + prompt_text as full context
                source_blob = (response_text or '') + '\n\n' + (prompt_text or '')

        final_path = None
        diff_path = None
        if final_code:
            # ensure final_code is just the code text
            code_text = unescape_text(final_code)
            if code_text is None:
                code_text = ''
            code_text = code_text.strip() + '\n'

            # try to extract surrounding diff/context by removing the code from the full blob
            diff_blob = None
            try:
                if source_blob and isinstance(source_blob, str):
                    # unescape source_blob similarly
                    sblob = unescape_text(source_blob) or source_blob
                    # if the code_text appears inside, remove it to get diff/context
                    if code_text.strip() and code_text.strip() in sblob:
                        diff_blob = sblob.replace(code_text.strip(), '').strip()
                    else:
                        # also try removing without trailing newline differences
                        if code_text.strip() and code_text.strip() in sblob.replace('\r\n','\n'):
                            diff_blob = sblob.replace(code_text.strip(), '').strip()
                        else:
                            # fallback: if source_blob contains more than just code, keep it as diff
                            if len(sblob) > len(code_text) + 50:
                                diff_blob = sblob
                else:
                    diff_blob = None
            except Exception:
                diff_blob = None

            # if code_text doesn't look like code, try to extract a code block from the full source_blob
            def looks_like_code(t: str) -> bool:
                return any(x in t for x in ('def ', 'class ', 'import ', 'from '))

            if not looks_like_code(code_text) and source_blob and isinstance(source_blob, str):
                sb = unescape_text(source_blob) or source_blob
                # try fenced code blocks first
                fenced = re.findall(r"```(?:python)?\n([\s\S]*?)```", sb, flags=re.I)
                if fenced:
                    # pick the largest fenced block that looks like code
                    cand = sorted(fenced, key=lambda s: len(s), reverse=True)[0]
                    if looks_like_code(cand):
                        code_text = cand.strip() + '\n'
                else:
                    # fallback: find first import/def/class and take onwards
                    m = re.search(r"(^|\n)(import\s.+|from\s.+|def\s.+|class\s.+)", sb)
                    if m:
                        start = m.start(2)
                        cand = sb[start:]
                        # if there are explicit markers like '###' or large markdown separators, chop them off
                        # keep until end; assume remaining text is code or code+comments
                        code_text = cand.strip() + '\n'

            # if still not found, try reading the original prompt JSON file (pf) and extract fenced code there
            if not looks_like_code(code_text) and pf and os.path.isfile(pf):
                try:
                    with open(pf, 'r', encoding='utf-8') as pfh_raw:
                        raw_text = pfh_raw.read()
                    raw_unesc = unescape_text(raw_text) or raw_text
                    fenced2 = re.findall(r"```(?:python)?\n([\s\S]*?)```", raw_unesc, flags=re.I)
                    if fenced2:
                        cand2 = sorted(fenced2, key=lambda s: len(s), reverse=True)[0]
                        if looks_like_code(cand2):
                            code_text = cand2.strip() + '\n'
                except Exception:
                    pass

            # write code file
            ext = '.py' if ('def ' in code_text or 'import ' in code_text or 'class ' in code_text) else '.txt'
            final_path = os.path.join(iter_dir, 'final_code' + ext)
            try:
                with open(final_path, 'w', encoding='utf-8') as cf:
                    cf.write(code_text)
            except Exception:
                final_path = None

            # if we found diff/context, write it to a separate file
            if diff_blob:
                diff_path = os.path.join(iter_dir, 'original_diff.txt')
                try:
                    with open(diff_path, 'w', encoding='utf-8') as df:
                        df.write(diff_blob.rstrip() + '\n')
                except Exception:
                    diff_path = None

        # Only record iteration and combined_score in the summary CSV (user request)
        rows.append({
            'iteration': it,
            'combined_score': combined,
        })

    # write CSV summary
    out_summary = os.path.join(out_base, 'iteration_summary.csv')
    fieldnames = ['iteration', 'combined_score']
    with open(out_summary, 'w', encoding='utf-8', newline='') as outf:
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(rows, key=lambda x: x['iteration']):
            # ensure only the requested columns are written
            writer.writerow({'iteration': r.get('iteration'), 'combined_score': r.get('combined_score')})
    # Also place a copy of the summary inside the exports directory requested
    try:
        export_summary_path = os.path.join(out_exports, 'iteration_summary.csv')
        shutil.copy(out_summary, export_summary_path)
    except Exception:
        # non-fatal: keep original summary and continue
        pass

    print(f'Wrote summary {out_summary} with {len(rows)} iterations')
    print(f'Wrote exports into {out_exports}')


if __name__ == '__main__':
    main()
