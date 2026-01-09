"""
Meta-prompt evolution support for OpenEvolve.
"""

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from openevolve.config import PromptConfig
from openevolve.utils.format_utils import format_metrics_safe
from openevolve.utils.metrics_utils import get_fitness_score

logger = logging.getLogger(__name__)


META_PROMPT_SYSTEM = (
    "You are an expert prompt engineer for code evolution. "
    "Write multiple concise bullet points that will be inserted into a user prompt. "
    "Each bullet should be an actionable instruction that helps improve code quality "
    "and evaluation score. Make the bullets specific to the task and primary metric. "
    "Use '-' bullets. "
    "Return ONLY the bullet list, without quotes or extra prefixes."
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

    def __init__(self, config: PromptConfig, llm_ensemble, language: str, system_message: str):
        self.config = config
        self.llm_ensemble = llm_ensemble
        self.language = language
        self.system_message = self._resolve_system_message(system_message)
        self.prompts: Dict[str, MetaPrompt] = {}
        self.archive: Dict[str, MetaPrompt] = {}

    def sample(self, parent_program, feature_dimensions: Optional[list] = None) -> MetaPrompt:
        """Sample a meta prompt, possibly generating a new one via LLM."""
        feature_dimensions = feature_dimensions or []

        should_generate = (
            not self.prompts or random.random() < self.config.meta_prompt_mutation_rate
        )
        if should_generate:
            seed = self._pick_seed_prompt()
            text = self._generate_meta_prompt(
                parent_program=parent_program,
                feature_dimensions=feature_dimensions,
                seed_prompt=seed.text if seed else None,
            )
            prompt = MetaPrompt(id=str(uuid.uuid4()), text=text, last_used=time.time())
            self._add(prompt)
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
    ) -> str:
        """Generate a meta prompt with the LLM."""
        parent_metrics = getattr(parent_program, "metrics", {}) or {}
        parent_fitness = get_fitness_score(parent_metrics, feature_dimensions)
        metrics_str = format_metrics_safe(parent_metrics)
        focus, domain_hints = self._infer_focus(parent_program, parent_metrics)

        user_msg = (
            "Task system message:\n"
            f"{self.system_message}\n\n"
            f"Programming language: {self.language}\n"
            f"Primary objective: maximize {focus}\n"
            f"Parent fitness: {parent_fitness:.4f}\n"
            f"Parent metrics: {metrics_str}\n"
        )
        if domain_hints:
            user_msg += f"\nDomain hints: {domain_hints}\n"
        if seed_prompt:
            user_msg += f"\nSeed instruction to refine:\n{seed_prompt}\n"
        user_msg += (
            "\nWrite a concise instruction (max "
            f"{self.config.meta_prompt_max_chars} chars) that will help improve the next "
            "code change. Focus on actionable guidance."
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

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.llm_ensemble.generate_with_context(
                            system_message=META_PROMPT_SYSTEM,
                            messages=[{"role": "user", "content": user_msg}],
                        ),
                    )
                    return future.result()
            except RuntimeError:
                return asyncio.run(
                    self.llm_ensemble.generate_with_context(
                        system_message=META_PROMPT_SYSTEM,
                        messages=[{"role": "user", "content": user_msg}],
                    )
                )
        except Exception as e:
            logger.warning(f"Meta-prompt LLM generation failed: {e}")
            return None

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
