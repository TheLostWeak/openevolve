"""
Meta prompt evolution utilities for OpenEvolve.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class MetaPromptEntry:
    """Represents an evolving meta prompt candidate."""

    id: str
    text: str
    score: float = 0.0
    uses: int = 0
    created_at: float = 0.0
    last_used: Optional[float] = None


class MetaPromptStore:
    """Simple JSON-backed store for meta prompts."""

    def __init__(self, path: Path, max_entries: int = 50):
        self.path = path
        self.max_entries = max_entries
        self.entries = self._load()

    def _load(self) -> Dict[str, MetaPromptEntry]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        entries = {}
        for item in data:
            entry = MetaPromptEntry(**item)
            entries[entry.id] = entry
        return entries

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(entry) for entry in self.entries.values()]
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temp_path.replace(self.path)

    def sample(self, rng, use_softmax: bool = False, temperature: float = 1.0) -> Optional[MetaPromptEntry]:
        if not self.entries:
            return None
        entries = list(self.entries.values())
        if use_softmax:
            temp = max(float(temperature), 1e-6)
            scores = [float(entry.score) for entry in entries]
            max_score = max(scores)
            weights = [pow(2.718281828, (s - max_score) / temp) for s in scores]
        else:
            weights = [max(entry.score, 0.0) + 1e-3 for entry in entries]
        if not any(weights):
            weights = [1.0] * len(entries)
        return rng.choices(entries, weights=weights, k=1)[0]

    def record_use(self, entry_id: str) -> None:
        entry = self.entries.get(entry_id)
        if not entry:
            return
        entry.uses += 1
        entry.last_used = time.time()

    def update_score(self, entry_id: str, delta: float) -> None:
        entry = self.entries.get(entry_id)
        if not entry:
            return
        entry.score += float(delta)

    def add(self, text: str, score: float) -> MetaPromptEntry:
        entry = MetaPromptEntry(
            id=uuid.uuid4().hex,
            text=text.strip(),
            score=float(score),
            uses=0,
            created_at=time.time(),
        )
        self.entries[entry.id] = entry
        if len(self.entries) > self.max_entries:
            # Drop the lowest scoring entry to keep store bounded.
            lowest = min(self.entries.values(), key=lambda e: e.score)
            self.entries.pop(lowest.id, None)
        return entry

    def flush(self) -> None:
        self._save()


def resolve_meta_prompt_path(meta_prompt_db_path: Optional[str], db_path: Optional[str]) -> Path:
    """Resolve the meta prompt storage path."""
    if meta_prompt_db_path:
        return Path(meta_prompt_db_path)
    if db_path:
        return Path(db_path) / "meta_prompts.json"
    return Path("meta_prompts.json")


def format_meta_prompt_block(text: str) -> str:
    if not text:
        return ""
    return "## Meta Prompt (evolving)\n" + text.strip() + "\n"


def build_meta_prompt_request(context: Dict[str, Any], max_chars: int) -> Tuple[str, str]:
    system = "You are a prompt engineer optimizing instructions for an automated code evolution loop."
    parent_metrics = context.get("parent_metrics", {})
    child_metrics = context.get("child_metrics", {})
    changes = context.get("changes_summary", "")
    error = context.get("error", "")
    llm_response = context.get("llm_response", "")
    system_message = context.get("system_message", "")

    def _truncate(text: str, limit: int) -> str:
        if not text:
            return ""
        return text[:limit] + ("..." if len(text) > limit else "")

    user = (
        "We run an evolutionary loop that edits code via LLM diffs and evaluates automatically.\n"
        "Propose 1-3 concise instruction lines to add to the prompt so future edits improve the score.\n"
        "Base your suggestion on the feedback below. Keep the meta prompt under "
        f"{max_chars} characters. Avoid generic advice.\n\n"
        f"Task system message (trimmed):\n{_truncate(system_message, 1200)}\n\n"
        f"Parent metrics: {parent_metrics}\n"
        f"Child metrics: {child_metrics}\n"
        f"Changes summary: {_truncate(changes, 600)}\n"
        f"LLM response (trimmed):\n{_truncate(llm_response, 800)}\n"
    )
    if error:
        user += f"\nEvaluation error: {_truncate(error, 400)}\n"
    user += "\nReturn only the meta prompt text."
    return system, user


def parse_meta_prompt_response(response: str, max_chars: int) -> str:
    text = response.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
    # Try JSON extraction if the model returns it.
    if text.startswith("{") and text.endswith("}"):
        try:
            data = json.loads(text)
            text = data.get("meta_prompt", text)
        except Exception:
            pass
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text
