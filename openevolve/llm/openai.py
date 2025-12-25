"""
OpenAI API interface for LLMs
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union

import openai

from openevolve.config import LLMConfig
from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class OpenAILLM(LLMInterface):
    """LLM interface using OpenAI-compatible APIs"""

    def __init__(
        self,
        model_cfg: Optional[dict] = None,
    ):
        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        # Allow env fallback when config leaves api_key empty
        self.api_base = model_cfg.api_base
        self.api_key = model_cfg.api_key or None
        self.random_seed = getattr(model_cfg, "random_seed", None)
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)

        # Set up API client
        # OpenAI client requires max_retries to be int, not None
        max_retries = self.retries if self.retries is not None else 0
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout,
            max_retries=max_retries,
        )

        # Only log unique models to reduce duplication
        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()

        if self.model not in logger._initialized_models:
            logger.info(f"Initialized OpenAI LLM with model: {self.model}")
            logger._initialized_models.add(self.model)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        # Prepare messages with system message
        formatted_messages = [{"role": "system", "content": system_message}]
        formatted_messages.extend(messages)

        # Set up generation parameters
        # Define OpenAI reasoning models that require max_completion_tokens
        # These models don't support temperature/top_p and use different parameters
        OPENAI_REASONING_MODEL_PREFIXES = (
            # O-series reasoning models
            "o1-",
            "o1",  # o1, o1-mini, o1-preview
            "o3-",
            "o3",  # o3, o3-mini, o3-pro
            "o4-",  # o4-mini
            # GPT-5 series are also reasoning models
            "gpt-5-",
            "gpt-5",  # gpt-5, gpt-5-mini, gpt-5-nano
            # The GPT OSS series are also reasoning models
            "gpt-oss-120b",
            "gpt-oss-20b",
        )

        # Check if this is an OpenAI reasoning model based on model name pattern
        # This works for all endpoints (OpenAI, Azure, OptiLLM, OpenRouter, etc.)
        model_lower = str(self.model).lower()
        is_openai_reasoning_model = model_lower.startswith(OPENAI_REASONING_MODEL_PREFIXES)

        if is_openai_reasoning_model:
            # For OpenAI reasoning models
            params = {
                "model": self.model,
                "messages": formatted_messages,
            }
            # Only include max_completion_tokens if explicitly configured.
            max_completion = kwargs.get("max_tokens", self.max_tokens)
            if max_completion is not None:
                params["max_completion_tokens"] = max_completion
            # Add optional reasoning parameters if provided
            reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
            if reasoning_effort is not None:
                params["reasoning_effort"] = reasoning_effort
            if "verbosity" in kwargs:
                params["verbosity"] = kwargs["verbosity"]
        else:
            # Standard parameters for all other models
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
            }
            # Only include max_tokens if explicitly configured.
            max_t = kwargs.get("max_tokens", self.max_tokens)
            if max_t is not None:
                params["max_tokens"] = max_t

            # Handle reasoning_effort for open source reasoning models.
            reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
            if reasoning_effort is not None:
                params["reasoning_effort"] = reasoning_effort

        # Support provider-specific extra body parameters (e.g., OpenRouter's extra_body)
        # Merge user-supplied extra_body into params so it is forwarded to the provider.
        user_extra = kwargs.get("extra_body")
        if user_extra is not None:
            # Ensure we don't overwrite an existing extra_body
            params["extra_body"] = params.get("extra_body", {})
            if isinstance(user_extra, dict):
                params["extra_body"].update(user_extra)
            else:
                # If extra_body provided as non-dict, just set it directly
                params["extra_body"] = user_extra

        # For reasoning models, ensure reasoning is enabled by default unless user overrides
        if is_openai_reasoning_model:
            eb = params.get("extra_body")
            if eb is None:
                params["extra_body"] = {"reasoning": {"enabled": True}}
            else:
                if isinstance(eb, dict):
                    eb.setdefault("reasoning", {"enabled": True})

        # Add seed parameter for reproducibility if configured
        # Skip seed parameter for Google AI Studio endpoint as it doesn't support it
        seed = kwargs.get("seed", self.random_seed)
        if seed is not None:
            if self.api_base == "https://generativelanguage.googleapis.com/v1beta/openai/":
                logger.warning(
                    "Skipping seed parameter as Google AI Studio endpoint doesn't support it. "
                    "Reproducibility may be limited."
                )
            else:
                params["seed"] = seed

        # Attempt the API call with retries
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(self._call_api(params), timeout=timeout)
                return response
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {str(e)}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {str(e)}")
                    raise

    async def _call_api(self, params: Dict[str, Any]) -> str:
        """Make the actual API call"""
        # Use asyncio to run the blocking API call in a thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self.client.chat.completions.create(**params)
        )
        # Logging of system prompt, user message and full response message (may include reasoning details)
        logger = logging.getLogger(__name__)
        logger.debug("API parameters: %s", params)
        try:
            message = response.choices[0].message
            # Log content and any reasoning details if present
            logger.debug("API response content: %s", getattr(message, "content", None))
            rd = getattr(message, "reasoning_details", None)
            if rd is not None:
                logger.debug("API response reasoning_details: %s", rd)

            content = getattr(message, "content", None)
            # If content is empty/None, dump full response repr for debugging to a debug folder
            if not content:
                try:
                    import os, time, uuid

                    debug_dir = os.path.join(os.getcwd(), "examples", "cap_set_example", "openevolve_output", "db", "llm_response_debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    fname = f"response_debug_{int(time.time())}_{uuid.uuid4().hex}.repr.txt"
                    fpath = os.path.join(debug_dir, fname)
                    with open(fpath, "w", encoding="utf-8") as fh:
                        fh.write("--- repr(response) ---\n")
                        try:
                            fh.write(repr(response))
                        except Exception:
                            try:
                                fh.write(str(response))
                            except Exception:
                                fh.write("<unrepresentable response object>")
                        fh.write("\n\n--- raw content attempt ---\n")
                        try:
                            # attempt to dump common fields
                            fh.write(str(getattr(response, "__dict__", {})))
                        except Exception:
                            pass
                    logger.info(f"Saved raw response repr to {fpath} for debugging")
                except Exception:
                    logger.debug("Failed to write response debug file")

            return content or ""
        except Exception:
            # Fallback: try to return raw text if structure differs
            logger.debug("API response (raw): %s", response)
            try:
                return str(response)
            except Exception:
                return ""
