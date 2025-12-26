"""
Utilities for code parsing, diffing, and manipulation
"""

import re
import unicodedata
import ast
import difflib
from typing import Dict, List, Optional, Tuple, Union


def parse_evolve_blocks(code: str) -> List[Tuple[int, int, str]]:
    """
    Parse evolve blocks from code

    Args:
        code: Source code with evolve blocks

    Returns:
        List of tuples (start_line, end_line, block_content)
    """
    lines = code.split("\n")
    blocks = []

    in_block = False
    start_line = -1
    block_content = []

    for i, line in enumerate(lines):
        if "# EVOLVE-BLOCK-START" in line:
            in_block = True
            start_line = i
            block_content = []
        elif "# EVOLVE-BLOCK-END" in line and in_block:
            in_block = False
            blocks.append((start_line, i, "\n".join(block_content)))
        elif in_block:
            block_content.append(line)

    return blocks


def apply_diff(
    original_code: str,
    diff_text: str,
    diff_pattern: str = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE",
) -> str:
    """
    Apply a diff to the original code

    Args:
        original_code: Original source code
        diff_text: Diff in the SEARCH/REPLACE format
        diff_pattern: Regex pattern for the SEARCH/REPLACE format

    Returns:
        Modified code
    """
    # Helper normalizer: unicode NFC, unify newlines, strip trailing whitespace
    def _normalize_text(s: str) -> str:
        s = unicodedata.normalize("NFC", s)
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        # strip trailing whitespace on each line
        s = "\n".join([ln.rstrip() for ln in s.split("\n")])
        return s

    def _collapse_whitespace(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    # Extract diff blocks
    diff_blocks = extract_diffs(diff_text, diff_pattern)

    # Work on a single string for easier AST-based replacement
    result_text = _normalize_text(original_code)

    for search_text, replace_text in diff_blocks:
        search_text_n = _normalize_text(search_text)
        replace_text_n = _normalize_text(replace_text)

        # 1) Try exact normalized match
        if search_text_n in result_text:
            result_text = result_text.replace(search_text_n, replace_text_n, 1)
            continue

        # 2) Try whitespace-collapsed exact match
        collapsed_result = _collapse_whitespace(result_text)
        collapsed_search = _collapse_whitespace(search_text_n)
        if collapsed_search in collapsed_result:
            # Find the best span in original result_text that corresponds to collapsed match
            # Use difflib to map
            seqmatcher = difflib.SequenceMatcher(a=collapsed_result, b=collapsed_search)
            match = seqmatcher.find_longest_match(0, len(collapsed_result), 0, len(collapsed_search))
            if match.size > 0:
                # Map collapsed_result span back to approximate location in result_text by searching substring
                sub = collapsed_result[match.a : match.a + match.size]
                idx = result_text.find(sub)
                if idx != -1:
                    result_text = result_text[:idx] + replace_text_n + result_text[idx + len(sub) :]
                    continue

        # 3) Try AST-aware replacement for Python named defs/classes
        ast_replaced = False
        try:
            # If search_text looks like a def/class header, extract the name
            m = re.match(r"^\s*(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", search_text_n)
            if m:
                kind, name = m.group(1), m.group(2)
                # Parse result_text AST and locate node
                module = ast.parse(result_text)
                for node in module.body:
                    if (isinstance(node, ast.FunctionDef) and kind == "def" and node.name == name) or (
                        isinstance(node, ast.ClassDef) and kind == "class" and node.name == name
                    ):
                        # Python 3.8+ exposes end_lineno
                        start = node.lineno - 1
                        end = getattr(node, "end_lineno", None)
                        if end is None:
                            # Fallback: approximate by searching next top-level node lineno
                            # Build list of top-level linenos
                            top_linenos = [n.lineno for n in module.body if hasattr(n, "lineno")]
                            top_linenos_sorted = sorted([ln for ln in top_linenos if ln > node.lineno])
                            end = (top_linenos_sorted[0] - 1) if top_linenos_sorted else len(result_text.split("\n"))
                        else:
                            end = end

                        orig_lines = result_text.split("\n")
                        # Replace lines start:end with replace_text_n lines
                        replace_lines = replace_text_n.split("\n")
                        result_lines = orig_lines[:start] + replace_lines + orig_lines[end:]
                        result_text = "\n".join(result_lines)
                        ast_replaced = True
                        break
        except Exception:
            ast_replaced = False

        if ast_replaced:
            continue

        # 4) Fuzzy character-level match using edit distance over windows of lines
        orig_lines = result_text.split("\n")
        search_lines = search_text_n.split("\n")
        best_match = None  # (score, start_idx, window_len)

        # consider window lengths between len(search_lines)-2 and +2
        min_len = max(1, len(search_lines) - 2)
        max_len = len(search_lines) + 2

        for wlen in range(min_len, max_len + 1):
            for i in range(0, len(orig_lines) - wlen + 1):
                window = "\n".join(orig_lines[i : i + wlen])
                dist = calculate_edit_distance(window, search_text_n)
                norm = dist / max(1, max(len(window), len(search_text_n)))
                if best_match is None or norm < best_match[0]:
                    best_match = (norm, i, wlen)

        if best_match and best_match[0] <= 0.20:
            # Accept replacement if normalized edit distance <= 0.20
            _, i, wlen = best_match
            orig_lines = result_text.split("\n")
            replace_lines = replace_text_n.split("\n")
            # Annotate fuzzy application with a comment for audit
            replace_block = ["# >>> Applied fuzzy SEARCH/REPLACE (score={:.3f})".format(best_match[0])] + replace_lines + ["# <<< End fuzzy applied block"]
            orig_lines[i : i + wlen] = replace_block
            result_text = "\n".join(orig_lines)
            continue

        # If we reach here, no replacement applied — leave unchanged and continue

    return result_text


def extract_diffs(
    diff_text: str, diff_pattern: str = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
) -> List[Tuple[str, str]]:
    """
    Extract diff blocks from the diff text

    Args:
        diff_text: Diff in the SEARCH/REPLACE format
        diff_pattern: Regex pattern for the SEARCH/REPLACE format

    Returns:
        List of tuples (search_text, replace_text)
    """
    diff_blocks = re.findall(diff_pattern, diff_text, re.DOTALL)
    return [(match[0].rstrip(), match[1].rstrip()) for match in diff_blocks]


def parse_full_rewrite(llm_response: str, language: str = "python") -> Optional[str]:
    """
    Extract a full rewrite from an LLM response

    Args:
        llm_response: Response from the LLM
        language: Programming language

    Returns:
        Extracted code or None if not found
    """
    # 1) Prefer explicit fenced code blocks with language tag, e.g. ```python
    code_block_pattern = r"```" + re.escape(language) + r"\r?\n(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL | re.IGNORECASE)

    if matches:
        return matches[0].strip()

    # 2) Fallback to any fenced code block (no language tag)
    code_block_pattern = r"```(?:\w+)?\r?\n(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # 3) Detect custom full-module markers used by some prompts: --- FULL MODULE START --- ... --- FULL MODULE END ---
    full_module_pattern = r"---\s*FULL MODULE START\s*---(.*?)(?:---\s*FULL MODULE END\s*---|$)"
    matches = re.findall(full_module_pattern, llm_response, re.DOTALL | re.IGNORECASE)
    if matches:
        # Inner content may itself contain fenced blocks; try to strip fences if present
        inner = matches[0].strip()
        # Remove leading/trailing triple-backticks if present
        inner = re.sub(r"^```(?:\w+)?\r?\n", "", inner)
        inner = re.sub(r"\r?\n```$", "", inner)
        return inner.strip()

    # 4) If the model returned a raw module without fences, try to heuristically extract
    # code by locating a useful code start (imports, module-level constants, or def/class)
    # and return from that point to the end (trimmed).
    heuristic_start = re.search(r"(^|\n)(?:from\s+[\w\.]+|import\s+[\w_,\s]+|_RANDOM_SEED|def\s+generate_set|class\s+\w+)", llm_response)
    if heuristic_start:
        start_idx = heuristic_start.start()
        candidate = llm_response[start_idx:].strip()
        # If candidate contains closing markers (like '--- FULL MODULE END ---'), strip them
        candidate = re.split(r"---\s*FULL MODULE END\s*---", candidate, flags=re.IGNORECASE)[0]
        # Limit size to avoid returning enormous unrelated text
        return candidate.strip()

    # 5) As a last resort, return the entire response (caller will validate)
    return llm_response.strip()


def format_diff_summary(diff_blocks: List[Tuple[str, str]]) -> str:
    """
    Create a human-readable summary of the diff

    Args:
        diff_blocks: List of (search_text, replace_text) tuples

    Returns:
        Summary string
    """
    summary = []

    for i, (search_text, replace_text) in enumerate(diff_blocks):
        search_lines = search_text.strip().split("\n")
        replace_lines = replace_text.strip().split("\n")

        # Create a short summary
        if len(search_lines) == 1 and len(replace_lines) == 1:
            summary.append(f"Change {i+1}: '{search_lines[0]}' to '{replace_lines[0]}'")
        else:
            search_summary = (
                f"{len(search_lines)} lines" if len(search_lines) > 1 else search_lines[0]
            )
            replace_summary = (
                f"{len(replace_lines)} lines" if len(replace_lines) > 1 else replace_lines[0]
            )
            summary.append(f"Change {i+1}: Replace {search_summary} with {replace_summary}")

    return "\n".join(summary)


def calculate_edit_distance(code1: str, code2: str) -> int:
    """
    Calculate the Levenshtein edit distance between two code snippets

    Args:
        code1: First code snippet
        code2: Second code snippet

    Returns:
        Edit distance (number of operations needed to transform code1 into code2)
    """
    if code1 == code2:
        return 0

    # Simple implementation of Levenshtein distance
    m, n = len(code1), len(code2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if code1[i - 1] == code2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return dp[m][n]


def extract_code_language(code: str) -> str:
    """
    Try to determine the language of a code snippet

    Args:
        code: Code snippet

    Returns:
        Detected language or "unknown"
    """
    # Look for common language signatures
    if re.search(r"^(import|from|def|class)\s", code, re.MULTILINE):
        return "python"
    elif re.search(r"^(package|import java|public class)", code, re.MULTILINE):
        return "java"
    elif re.search(r"^(#include|int main|void main)", code, re.MULTILINE):
        return "cpp"
    elif re.search(r"^(function|var|let|const|console\.log)", code, re.MULTILINE):
        return "javascript"
    elif re.search(r"^(module|fn|let mut|impl)", code, re.MULTILINE):
        return "rust"
    elif re.search(r"^(SELECT|CREATE TABLE|INSERT INTO)", code, re.MULTILINE):
        return "sql"

    return "unknown"
