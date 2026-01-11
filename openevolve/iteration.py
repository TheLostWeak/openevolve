import asyncio
import os
import uuid
import logging
import time
import random
from dataclasses import dataclass

from openevolve.database import Program, ProgramDatabase
from openevolve.config import Config
from openevolve.evaluator import Evaluator
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.prompt.sampler import PromptSampler
from openevolve.prompt.meta_prompt import (
    MetaPromptStore,
    resolve_meta_prompt_path,
    format_meta_prompt_block,
    build_meta_prompt_request,
    parse_meta_prompt_response,
)
from openevolve.utils.code_utils import (
    apply_diff,
    extract_diffs,
    format_diff_summary,
    parse_full_rewrite,
)
from openevolve.utils.metrics_utils import get_fitness_score, is_valid_metrics


@dataclass
class Result:
    """Resulting program and metrics from an iteration of OpenEvolve"""

    child_program: str = None
    parent: str = None
    child_metrics: str = None
    iteration_time: float = None
    prompt: str = None
    llm_response: str = None
    artifacts: dict = None


# Global store for reasoning message history (reasoning_details chaining)
_reasoning_message_history = []


def get_reasoning_message_history():
    """Get the global reasoning message history for chaining."""
    return _reasoning_message_history


def reset_reasoning_message_history():
    """Reset the reasoning message history (e.g., between sessions)."""
    global _reasoning_message_history
    _reasoning_message_history = []


async def run_iteration_with_shared_db(
    iteration: int,
    config: Config,
    database: ProgramDatabase,
    evaluator: Evaluator,
    llm_ensemble: LLMEnsemble,
    prompt_sampler: PromptSampler,
):
    """
    Run a single iteration using shared memory database

    This is optimized for use with persistent worker processes.
    """
    logger = logging.getLogger(__name__)

    try:
        # Sample parent and inspirations from database
        parent, inspirations = database.sample(num_inspirations=config.prompt.num_top_programs)

        # Get artifacts for the parent program if available
        parent_artifacts = database.get_artifacts(parent.id)

        # Get island-specific top programs for prompt context (maintain island isolation)
        parent_island = parent.metadata.get("island", database.current_island)
        island_top_programs = database.get_top_programs(5, island_idx=parent_island)
        island_previous_programs = database.get_top_programs(3, island_idx=parent_island)

        # Build prompt
        meta_prompt_id = None
        meta_prompt_text = ""
        meta_prompt_store = None
        if config.prompt.use_meta_prompting:
            meta_prompt_path = resolve_meta_prompt_path(
                config.prompt.meta_prompt_db_path, database.config.db_path
            )
            meta_prompt_store = MetaPromptStore(
                meta_prompt_path, max_entries=config.prompt.meta_prompt_max_entries
            )
            rng = random.Random()
            if config.random_seed is not None:
                rng.seed(config.random_seed + iteration)
            if rng.random() < config.prompt.meta_prompt_weight:
                entry = meta_prompt_store.sample(
                    rng,
                    use_softmax=config.prompt.meta_prompt_use_softmax,
                    temperature=config.prompt.meta_prompt_temperature,
                )
                if entry:
                    meta_prompt_id = entry.id
                    meta_prompt_text = format_meta_prompt_block(entry.text)

        prompt = prompt_sampler.build_prompt(
            current_program=parent.code,
            parent_program=parent.code,
            program_metrics=parent.metrics,
            previous_programs=[p.to_dict() for p in island_previous_programs],
            top_programs=[p.to_dict() for p in island_top_programs],
            inspirations=[p.to_dict() for p in inspirations],
            language=config.language,
            evolution_round=iteration,
            diff_based_evolution=config.diff_based_evolution,
            program_artifacts=parent_artifacts if parent_artifacts else None,
            feature_dimensions=database.config.feature_dimensions,
            meta_prompt=meta_prompt_text,
        )
        prompt["meta_prompt"] = meta_prompt_text
        if meta_prompt_id:
            prompt["meta_prompt_id"] = meta_prompt_id
            prompt["meta_prompt_text"] = meta_prompt_text.strip()

        result = Result(parent=parent)
        iteration_start = time.time()

        # Build messages list with reasoning history for continuous reasoning
        messages = [{"role": "user", "content": prompt["user"]}]
        
        # Prepend any previous reasoning messages (for continuous reasoning chaining)
        reasoning_history = get_reasoning_message_history()
        if reasoning_history:
            messages = reasoning_history + messages

        # Generate code modification with return_message=True to get reasoning_details
        llm_response_obj = await llm_ensemble.generate_with_context(
            system_message=prompt["system"],
            messages=messages,
            return_message=True,  # Enable to get reasoning_details for chaining
        )
        
        # Extract content and handle both dict (with reasoning) and string responses
        if isinstance(llm_response_obj, dict):
            llm_response = llm_response_obj.get("content", "")
            reasoning_details = llm_response_obj.get("reasoning_details")
            
            if reasoning_details is not None:
                # Store this response as part of reasoning history for next iteration
                assistant_msg = {
                    "role": "assistant",
                    "content": llm_response,
                    "reasoning_details": reasoning_details,
                }
                # Keep last N messages to avoid unbounded growth; keep recent reasoning
                global _reasoning_message_history
                _reasoning_message_history.append(assistant_msg)
                # Optionally limit history to last 5 turns to manage size
                if len(_reasoning_message_history) > 5:
                    _reasoning_message_history = _reasoning_message_history[-5:]
        else:
            # Fallback: response is plain string (no reasoning or return_message=False)
            llm_response = llm_response_obj

        # Parse the response
        if config.diff_based_evolution:
            diff_blocks = extract_diffs(llm_response, config.diff_pattern)

            if not diff_blocks:
                logger.warning(f"Iteration {iteration+1}: No valid diffs found in response")
                return None

            # Apply the diffs
            child_code = apply_diff(parent.code, llm_response, config.diff_pattern)
            changes_summary = format_diff_summary(diff_blocks)
        else:
            # Parse full rewrite
            new_code = parse_full_rewrite(llm_response, config.language)

            if not new_code:
                logger.warning(f"Iteration {iteration+1}: No valid code found in response")
                return None

            child_code = new_code
            changes_summary = "Full rewrite"

        # Check code length
        if len(child_code) > config.max_code_length:
            logger.warning(
                f"Iteration {iteration+1}: Generated code exceeds maximum length "
                f"({len(child_code)} > {config.max_code_length})"
            )
            return None

        # Evaluate the child program
        child_id = str(uuid.uuid4())
        result.child_metrics = await evaluator.evaluate_program(child_code, child_id)

        # Handle artifacts if they exist
        artifacts = evaluator.get_pending_artifacts(child_id)

        # Set template_key of Prompts
        template_key = "full_rewrite_user" if not config.diff_based_evolution else "diff_user"

        # Create a child program
        result.child_program = Program(
            id=child_id,
            code=child_code,
            language=config.language,
            parent_id=parent.id,
            generation=parent.generation + 1,
            metrics=result.child_metrics,
            iteration_found=iteration,
            metadata={
                "changes": changes_summary,
                "parent_metrics": parent.metrics,
            },
            prompts=(
                {
                    template_key: {
                        "system": prompt["system"],
                        "user": prompt["user"],
                        "responses": [llm_response] if llm_response is not None else [],
                    }
                }
                if database.config.log_prompts
                else None
            ),
        )

        result.prompt = prompt
        result.llm_response = llm_response
        result.artifacts = artifacts
        result.iteration_time = time.time() - iteration_start
        result.iteration = iteration

        if meta_prompt_store and config.prompt.use_meta_prompting:
            if not is_valid_metrics(result.child_metrics):
                return result
            parent_fitness = get_fitness_score(
                parent.metrics, database.config.feature_dimensions
            )
            child_fitness = get_fitness_score(
                result.child_metrics, database.config.feature_dimensions
            )
            if meta_prompt_id:
                meta_prompt_store.update_score(meta_prompt_id, child_fitness - parent_fitness)
                meta_prompt_store.record_use(meta_prompt_id)

            if (
                config.prompt.meta_prompt_update_every > 0
                and iteration % config.prompt.meta_prompt_update_every == 0
            ):
                system_message, user_message = build_meta_prompt_request(
                    {
                        "system_message": prompt.get("system", ""),
                        "parent_metrics": parent.metrics,
                        "child_metrics": result.child_metrics,
                        "changes_summary": changes_summary,
                        "llm_response": llm_response,
                        "error": result.child_metrics.get("error") if result.child_metrics else "",
                    },
                    max_chars=config.prompt.meta_prompt_max_chars,
                )
                meta_response = await llm_ensemble.generate_with_context(
                    system_message=system_message,
                    messages=[{"role": "user", "content": user_message}],
                )
                candidate = parse_meta_prompt_response(
                    meta_response, config.prompt.meta_prompt_max_chars
                )
                if candidate:
                    meta_prompt_store.add(candidate, child_fitness)

            meta_prompt_store.flush()

        return result

    except Exception as e:
        logger.exception(f"Error in iteration {iteration}: {e}")
        return None
