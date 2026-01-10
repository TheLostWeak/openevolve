"""
Meta-prompt evolution support for OpenEvolve.
"""

import asyncio
import logging
import random
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from openevolve.config import PromptConfig
from openevolve.prompt.templates import TemplateManager
from openevolve.utils.format_utils import format_metrics_safe
from openevolve.utils.metrics_utils import get_fitness_score

logger = logging.getLogger(__name__)


META_PROMPT_SYSTEM = (
    "You are an expert meta-prompt engineer for iterative code evolution. You should generate a "
    "COMPLETE, standalone replacement meta-prompt that will be used to guide the next code change. "
    "The replacement should be concise, actionable, and specific to the task and primary metric. "
    "Prioritize suggestions that target measurable improvement while keeping changes minimal and safe. "
    "Do NOT include explanations, rationale, or any prefixes — return ONLY the new prompt text."
)


@dataclass
class MetaPrompt:
    """A single meta prompt candidate."""

    id: str
    text: str
    score: float = 0.0
    count: int = 0
    last_used: float = 0.0


class MetaPromptDatabase:
    """Manage meta prompts and evolve them using LLM feedback."""

    def __init__(
        self,
        config: PromptConfig,
        llm_ensemble,
        language: str,
        system_message: str,
        initial_user_template_key: Optional[str] = None,
    ):
        self.config = config
        self.llm_ensemble = llm_ensemble
        self.language = language
        self.system_message = self._resolve_system_message(system_message)
        self.prompts: Dict[str, MetaPrompt] = {}
        self.archive: Dict[str, MetaPrompt] = {}

        # Track current user template (key + body) used for code generation.
        self.template_manager = TemplateManager(custom_template_dir=self.config.template_dir)
        self.current_user_template_key = initial_user_template_key or "full_rewrite_user"
        try:
            self.current_user_template_body = self.template_manager.get_template(
                self.current_user_template_key
            )
        except Exception:
            self.current_user_template_body = ""

    def sample(self, parent_program, feature_dimensions: Optional[list] = None) -> MetaPrompt:
        """Sample a meta prompt, possibly generating a new one via LLM."""
        feature_dimensions = feature_dimensions or []
        should_generate = (
            not self.prompts or random.random() < self.config.meta_prompt_mutation_rate
        )
        if should_generate:
            # collect richer context for LLM: seed prompt, current prompt text (if any),
            # parent code, previous code (if parent_program exposes it), fitness delta
            seed = self._pick_seed_prompt()

            # Try to obtain current prompt text from parent_program if available
            current_prompt_text = getattr(parent_program, "prompt", None) or getattr(
                parent_program, "meta_prompt", None
            )

            # current code and previous code (best-effort from parent_program attributes)
            current_code = getattr(parent_program, "code", "") or ""
            previous_code = getattr(parent_program, "previous_code", "") or ""

            # If caller attached a last_fitness_delta to parent_program, use it; else 0.0
            fitness_delta = getattr(parent_program, "last_fitness_delta", None)
            if fitness_delta is None:
                # fallback: if parent has metrics and parent has a 'previous_metrics'
                prev_metrics = getattr(parent_program, "previous_metrics", None)
                if prev_metrics is not None:
                    try:
                        cur = get_fitness_score(getattr(parent_program, "metrics", {}) or {}, feature_dimensions)
                        prev = get_fitness_score(prev_metrics or {}, feature_dimensions)
                        fitness_delta = cur - prev
                    except Exception:
                        fitness_delta = 0.0
                else:
                    fitness_delta = 0.0

            # collect a small set of historical prompts (archive best + some random for diversity)
            historical = []
            # include top archive entries first
            for p in list(self.archive.values()):
                historical.append(p.text)
            # if not enough, add random prompts for diversity
            if len(historical) < self.config.meta_prompt_history_size:
                needed = self.config.meta_prompt_history_size - len(historical)
                all_texts = [p.text for p in self.prompts.values() if p.text not in historical]
                if all_texts:
                    historical.extend(random.sample(all_texts, min(needed, len(all_texts))))

            text = self._generate_meta_prompt(
                parent_program=parent_program,
                feature_dimensions=feature_dimensions,
                seed_prompt=seed.text if seed else None,
                current_prompt_text=current_prompt_text,
                current_code=current_code,
                previous_code=previous_code,
                fitness_delta=fitness_delta,
                historical_prompts=historical,
            )
            prompt = MetaPrompt(id=str(uuid.uuid4()), text=text, last_used=time.time())
            self._add(prompt)
            # Replace current user template body with the newly generated meta-prompt (complete template)
            self.current_user_template_body = text
            # Mark key as meta-generated
            self.current_user_template_key = "meta_prompt_user"
            return prompt

        prompt = self._sample_from_archive() or self._sample_any()
        prompt.last_used = time.time()
        return prompt

    def update_score(self, prompt_id: str, fitness_delta: float) -> None:
        """Update a meta prompt's score using fitness delta."""
        prompt = self.prompts.get(prompt_id)
        if not prompt:
            return
        weighted_delta = fitness_delta * self.config.meta_prompt_weight
        prompt.count += 1
        # Cumulative average keeps scores comparable over time.
        prompt.score = ((prompt.score * (prompt.count - 1)) + weighted_delta) / prompt.count
        self._update_archive()
        self._enforce_population_limit()

    def _add(self, prompt: MetaPrompt) -> None:
        self.prompts[prompt.id] = prompt
        self._update_archive()
        self._enforce_population_limit()

    def _enforce_population_limit(self) -> None:
        if len(self.prompts) <= self.config.meta_prompt_population_size:
            return
        # Remove lowest-scoring prompts first.
        sorted_prompts = sorted(self.prompts.values(), key=lambda p: p.score)
        to_remove = len(self.prompts) - self.config.meta_prompt_population_size
        for prompt in sorted_prompts[:to_remove]:
            self.prompts.pop(prompt.id, None)
            self.archive.pop(prompt.id, None)

    def _update_archive(self) -> None:
        if not self.prompts:
            self.archive = {}
            return
        sorted_prompts = sorted(self.prompts.values(), key=lambda p: p.score, reverse=True)
        keep = sorted_prompts[: self.config.meta_prompt_archive_size]
        self.archive = {prompt.id: prompt for prompt in keep}

    def _pick_seed_prompt(self) -> Optional[MetaPrompt]:
        if not self.archive:
            return None
        return random.choice(list(self.archive.values()))

    def _sample_from_archive(self) -> Optional[MetaPrompt]:
        if not self.archive:
            return None
        prompts = list(self.archive.values())
        weights = [max(p.score, 0.0) + 1e-3 for p in prompts]
        return random.choices(prompts, weights=weights, k=1)[0]

    def _sample_any(self) -> MetaPrompt:
        return random.choice(list(self.prompts.values()))

    def _generate_meta_prompt(
        self,
        parent_program,
        feature_dimensions: list,
        seed_prompt: Optional[str] = None,
        current_prompt_text: Optional[str] = None,
        current_code: str = "",
        previous_code: str = "",
        fitness_delta: float = 0.0,
        historical_prompts: Optional[list] = None,
    ) -> str:
        """Generate a meta prompt with the LLM."""
        parent_metrics = getattr(parent_program, "metrics", {}) or {}
        # Remove any file/path/artifact related keys from metrics before sending to LLM
        sanitized_metrics = dict(parent_metrics)
        for k in list(sanitized_metrics.keys()):
            try:
                if "path" in k.lower() or "artifact" in k.lower():
                    sanitized_metrics.pop(k, None)
            except Exception:
                # on unexpected types, skip removal for that key
                continue

        parent_fitness = get_fitness_score(parent_metrics, feature_dimensions)
        metrics_str = format_metrics_safe(sanitized_metrics)

        # Attempt to fetch the original prompt template (with placeholders) without formatting.
        # Use TemplateManager to locate template by name if available; otherwise fall back to the
        # raw system message string provided when the database was constructed.
        try:
            # Resolve system template body
            original_template = None
            raw_candidate = getattr(self, "system_message", None)
            if isinstance(raw_candidate, str) and raw_candidate in self.template_manager.templates:
                original_template = self.template_manager.get_template(raw_candidate)
            else:
                original_template = self.system_message

            # Use tracked current user template
            user_template_key = self.current_user_template_key or "unknown"
            user_template_body = self.current_user_template_body or ""
        except Exception:
            original_template = self.system_message
            user_template_key = getattr(self, "current_user_template_key", "unknown")
            user_template_body = getattr(self, "current_user_template_body", "")
        focus, domain_hints = self._infer_focus(parent_program, parent_metrics)
        # Include the original prompt/system template explicitly as context for refinement,
        # but do NOT rely on it as the LLM system role. The actual system role used is
        # the dedicated META_PROMPT_SYSTEM passed when calling the LLM.
        # Use explicit separators to avoid LLM confusion between sections
        user_msg = "".join([
            "---\n",
            "Meta-prompt design hints:\n",
            f"{META_PROMPT_SYSTEM}\n",
            "---\n",
            "Original prompt template (KEEP placeholders like {metrics}, {current_program}, etc.; do NOT substitute):\n",
            f"{original_template}\n",
            "---\n",
            f"Active user template key: {user_template_key}\n",
            (f"Active user template body:\n{user_template_body}\n" if user_template_body else ""),
            "---\n",
            f"Programming language: {self.language}\n",
            f"Primary objective: maximize {focus}\n",
            f"Parent fitness (current): {parent_fitness:.4f}\n",
            f"Parent metrics (sanitized): {metrics_str}\n",
            "---\n",
        ])
        if domain_hints:
            user_msg += f"\nDomain hints: {domain_hints}\n"

        # Provide richer context: current prompt, seed, code before/after, fitness_delta, history
        if current_prompt_text:
            user_msg += f"\nCurrent meta-prompt (seed instruction/template to be replaced):\n{current_prompt_text}\n"
        if seed_prompt:
            user_msg += f"\nSeed instruction to refine:\n{seed_prompt}\n"
        if fitness_delta is not None:
            user_msg += f"\nLast fitness delta for current prompt: {fitness_delta:.6f}\n"
        if previous_code:
            user_msg += "".join(["Code before last prompt (previous_code):\n", previous_code, "\n", "---\n"])
        if current_code:
            user_msg += "".join(["Code after last prompt (current_code):\n", current_code, "\n", "---\n"])

        # Provide explicit score comparison if previous metrics exist
        prev_metrics = getattr(parent_program, "previous_metrics", None)
        if prev_metrics is not None:
            try:
                prev_fitness = get_fitness_score(prev_metrics or {}, feature_dimensions)
                user_msg += f"Previous fitness (before last prompt): {prev_fitness:.4f}\n"
                user_msg += f"Fitness delta (current - previous): {parent_fitness - prev_fitness:.4f}\n"
                user_msg += "---\n"
            except Exception:
                # ignore if fitness can't be computed
                pass
        else:
            # If no previous_metrics, include provided fitness_delta if available
            try:
                user_msg += f"Observed fitness delta for last prompt: {float(fitness_delta):.6f}\n"
                user_msg += "---\n"
            except Exception:
                pass
        if historical_prompts:
            user_msg += "Historical meta-prompts (examples):\n"
            for hp in historical_prompts[: self.config.meta_prompt_history_size]:
                user_msg += f"---\n{hp}\n"
            user_msg += "---\n"

        user_msg += (
            "Write a COMPLETE, standalone replacement prompt (max "
            f"{self.config.meta_prompt_max_chars} chars) that WILL REPLACE the entire current prompt/template.\n"
            "Return ONLY the new full prompt text — NO explanations, NO analysis, NO prefixes.\n"
            "The returned prompt must be a complete template that can be used directly as the task prompt for the next code generation step.\n"
            "---\n"
        )

        content = self._run_llm_sync(user_msg)
        if not content:
            return seed_prompt or "Focus on improving the evaluation score while keeping changes minimal."

        text = content.strip()
        if len(text) > self.config.meta_prompt_max_chars:
            text = text[: self.config.meta_prompt_max_chars].rstrip()
        return text

    def _run_llm_sync(self, user_msg: str) -> Optional[str]:
        """Run LLM call in a sync context, handling event loops."""
        try:
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures

                # Sanitize paths before logging to avoid recording local absolute paths
                sanitized = self._sanitize_for_logging(user_msg)
                logger.info("Meta-prompt API request (user_msg):\n%s", sanitized)

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.llm_ensemble.generate_with_context(
                            system_message=META_PROMPT_SYSTEM,
                            messages=[{"role": "user", "content": user_msg}],
                        ),
                    )
                    content = future.result()
                    logger.info(
                        "Meta-prompt API response (prompt template):\n%s",
                        self._sanitize_for_logging(str(content)),
                    )
                    return content
            except RuntimeError:
                # No running event loop; call directly and log
                sanitized = self._sanitize_for_logging(user_msg)
                logger.info("Meta-prompt API request (user_msg):\n%s", sanitized)
                content = asyncio.run(
                    self.llm_ensemble.generate_with_context(
                        system_message=META_PROMPT_SYSTEM,
                        messages=[{"role": "user", "content": user_msg}],
                    )
                )
                logger.info(
                    "Meta-prompt API response (prompt template):\n%s",
                    self._sanitize_for_logging(str(content)),
                )
                return content
        except Exception as e:
            logger.warning(f"Meta-prompt LLM generation failed: {e}")
            return None

    def _sanitize_for_logging(self, text: str) -> str:
        """Redact absolute file paths (Windows and Unix-like) to avoid leaking local paths in logs."""
        if not text:
            return text
        # Windows-style paths: C:\\Users\\... or D:\\path\\to\\file
        windows_pat = r"[A-Za-z]:\\\\(?:[^\\\\\n]+\\\\)*[^\\\\\n\\s]+"
        # Unix-style paths: /tmp/... or /home/user/...
        unix_pat = r"/(?:[^/\\n]+/)*[^/\\n\\s]+"
        redacted = re.sub(windows_pat, "[REDACTED_PATH]", text)
        redacted = re.sub(unix_pat, "[REDACTED_PATH]", redacted)
        return redacted

    def _resolve_system_message(self, system_message: str) -> str:
        """Resolve template names to actual system message text."""
        from openevolve.prompt.templates import TemplateManager

        template_manager = TemplateManager(custom_template_dir=self.config.template_dir)
        if system_message in template_manager.templates:
            return template_manager.get_template(system_message)
        return system_message

    def _infer_focus(self, parent_program, parent_metrics: Dict[str, Any]) -> Tuple[str, str]:
        """Infer the primary objective and optional domain hints."""
        code = getattr(parent_program, "code", "") or ""
        system_lower = (self.system_message or "").lower()
        code_lower = code.lower()

        if "cap set" in system_lower or "capset" in system_lower or "cap set" in code_lower:
            return (
                "cap set size (maximize size while preserving validity)",
                "Consider symmetry breaking, pruning invalid triples early, and faster set membership checks.",
            )

        if "size" in parent_metrics:
            return ("size metric", "")

        if "combined_score" in parent_metrics:
            return ("combined_score", "")

        return ("overall evaluation score", "")
