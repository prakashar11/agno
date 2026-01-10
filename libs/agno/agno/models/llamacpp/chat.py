"""
llama.cpp Server Model Wrapper for Agno

Provides an OpenAI-compatible interface to llama.cpp server (llama-server).
Supports multi-model routing, tool calling, and JSON schema constraints.

Drop-in Replacement for Ollama:
    This wrapper provides the same interface as agno's Ollama wrapper:
    - invoke(messages, ...) -> response with response.message.content (Ollama-compatible)
    - ainvoke(messages, ...) -> async response with response.message.content
    - invoke_stream(messages, ...) -> streaming response
    - ainvoke_stream(messages, ...) -> async streaming response
    - parse_provider_response(response) -> ModelResponse
    - parse_provider_response_delta(response) -> ModelResponse (streaming)

    Agent code using Ollama can switch to LlamaCpp without any changes.

    Response Compatibility:
        LlamaCpp wraps OpenAI responses to provide Ollama-style access:
        - response.message.content  -> text content
        - response.message.tool_calls -> tool calls (if any)

        This means code like `response.message.content` works identically
        for both Ollama and LlamaCpp.

Requirements:
    - llama-server running with --jinja flag for tool calling support
    - OpenAI Python client (pip install openai)

Example:
    from agno.models.llamacpp import LlamaCpp
    from agno.agent import Agent

    # Create LlamaCpp instance (same pattern as Ollama)
    llm = LlamaCpp(
        id="qwen2.5-7b",
        base_url="http://localhost:8080/v1",
        temperature=0.2,
    )

    # Use in Agent (identical to Ollama usage)
    agent = Agent(model=llm, tools=[my_tool])
    agent.run("Do something")

    # Direct invocation also works identically to Ollama
    response = llm.invoke([Message(role="user", content="Hello")])
    print(response.message.content)  # Works just like Ollama!

Multi-model routing:
    # Start llama-server with multiple models (preset file)
    # Then route by model name:
    llm_utility = LlamaCpp(id="qwen2.5-7b", base_url="http://localhost:8080/v1")
    llm_analysis = LlamaCpp(id="gpt-oss-20b", base_url="http://localhost:8080/v1")
"""

from dataclasses import dataclass, field
from os import getenv
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional, Type, Union

from pydantic import BaseModel

from agno.models.message import Message
from agno.models.openai.like import OpenAILike
from agno.models.response import ModelResponse
from agno.utils.log import log_debug


# =============================================================================
# OLLAMA-COMPATIBLE RESPONSE WRAPPERS
# =============================================================================
# These classes wrap OpenAI responses to provide Ollama-style attribute access.
# This enables code like `response.message.content` to work with LlamaCpp.


class OllamaCompatibleMessage:
    """
    Wrapper to provide Ollama-style message access from OpenAI response.

    Ollama returns: response.message.content
    OpenAI returns: response.choices[0].message.content

    This wrapper provides both access patterns.
    """

    def __init__(self, openai_message):
        """
        Args:
            openai_message: The message object from OpenAI's ChatCompletion.choices[0].message
        """
        self._openai_message = openai_message

    @property
    def content(self) -> Optional[str]:
        """Get message content (Ollama-compatible)."""
        return self._openai_message.content

    @property
    def role(self) -> str:
        """Get message role."""
        return self._openai_message.role

    @property
    def tool_calls(self) -> Optional[List[Any]]:
        """Get tool calls if present."""
        return getattr(self._openai_message, "tool_calls", None)

    def __getattr__(self, name):
        """Forward any other attribute access to underlying message."""
        return getattr(self._openai_message, name)

    def __repr__(self):
        return f"OllamaCompatibleMessage(role={self.role}, content={self.content[:50] if self.content else None}...)"


class OllamaCompatibleResponse:
    """
    Wrapper to provide Ollama-style response access from OpenAI ChatCompletion.

    Provides:
        response.message.content  (Ollama-style)
        response.choices[0].message.content  (OpenAI-style, also works)

    This enables code written for Ollama to work unchanged with LlamaCpp.
    """

    def __init__(self, openai_response):
        """
        Args:
            openai_response: OpenAI ChatCompletion response object
        """
        self._openai_response = openai_response
        # Wrap the first choice's message for Ollama-compatible access
        if openai_response.choices:
            self._message = OllamaCompatibleMessage(openai_response.choices[0].message)
        else:
            self._message = None

    @property
    def message(self) -> Optional[OllamaCompatibleMessage]:
        """Get message in Ollama format (response.message.content)."""
        return self._message

    @property
    def choices(self):
        """Get choices in OpenAI format (for compatibility)."""
        return self._openai_response.choices

    @property
    def model(self) -> str:
        """Get model name."""
        return self._openai_response.model

    @property
    def usage(self):
        """Get usage statistics."""
        return self._openai_response.usage

    def __getattr__(self, name):
        """Forward any other attribute access to underlying response."""
        return getattr(self._openai_response, name)

    def __repr__(self):
        content_preview = self._message.content[:50] if self._message and self._message.content else None
        return f"OllamaCompatibleResponse(model={self.model}, content={content_preview}...)"


@dataclass
class LlamaCpp(OpenAILike):
    """
    llama.cpp server wrapper using OpenAI-compatible API.

    Attributes:
        id: Model name/alias (used for routing in multi-model setups)
        name: Display name for this model instance
        provider: Provider identifier
        base_url: llama-server URL (default: http://localhost:8080/v1)
        api_key: API key (default: "not-required" as llama.cpp doesn't require auth)

    llama.cpp specific:
        cache_prompt: Enable KV cache reuse for repeated prompts
        slot_id: Target specific slot (for dedicated sessions)

    Performance tuning (set when starting llama-server):
        --ctx-size: Context window size
        --n-gpu-layers: GPU layer offloading
        --flash-attn: Flash attention
        --cache-type-k/v: KV cache quantization
        --parallel: Number of concurrent slots
        --cont-batching: Continuous batching for throughput
    """

    id: str = "not-set"
    name: str = "LlamaCpp"
    provider: str = "LlamaCpp"

    # Server configuration
    base_url: str = getenv("LLAMACPP_BASE_URL", "http://localhost:8080/v1")
    api_key: Optional[str] = getenv("LLAMACPP_API_KEY", "not-required")

    # llama.cpp doesn't support native structured outputs like OpenAI
    # but it does support JSON schema via grammar constraints
    supports_native_structured_outputs: bool = False
    supports_json_schema_outputs: bool = True

    # llama.cpp specific options (sent in extra_body)
    cache_prompt: Optional[bool] = None  # Enable prompt caching
    slot_id: Optional[int] = None  # Target specific slot for session affinity

    # Sampling parameters (can also be set on parent class)
    # These override parent class defaults for llama.cpp optimization
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None  # llama.cpp supports top_k
    repeat_penalty: Optional[float] = None  # Repetition penalty

    # Role mapping (llama.cpp uses standard OpenAI roles)
    default_role_map = {
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        "tool": "tool",
    }

    def __post_init__(self):
        """Validate configuration after initialization."""
        super().__post_init__()

        if self.id == "not-set":
            log_debug(
                "LlamaCpp model id not set. Make sure to specify the model name "
                "that matches your llama-server configuration."
            )

    def get_request_params(
        self,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Build request parameters for llama.cpp server.

        Extends parent class to add llama.cpp-specific options in extra_body.
        """
        # Get base parameters from OpenAILike
        request_kwargs = super().get_request_params(
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
        )

        # Build llama.cpp-specific extra_body parameters
        extra_body: Dict[str, Any] = request_kwargs.get("extra_body", {})

        # Add cache_prompt if specified
        if self.cache_prompt is not None:
            extra_body["cache_prompt"] = self.cache_prompt

        # Add slot_id for session affinity
        if self.slot_id is not None:
            extra_body["slot_id"] = self.slot_id

        # Add top_k if specified (not in standard OpenAI API)
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k

        # Add repeat_penalty if specified
        if self.repeat_penalty is not None:
            extra_body["repeat_penalty"] = self.repeat_penalty

        # Only add extra_body if we have llama.cpp-specific options
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        if request_kwargs:
            log_debug(
                f"Calling {self.provider} with request parameters: {request_kwargs}",
                log_level=2,
            )

        return request_kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model configuration to a dictionary."""
        model_dict = super().to_dict()
        model_dict.update(
            {
                "base_url": self.base_url,
                "cache_prompt": self.cache_prompt,
                "slot_id": self.slot_id,
                "top_k": self.top_k,
                "repeat_penalty": self.repeat_penalty,
            }
        )
        # Remove None values
        return {k: v for k, v in model_dict.items() if v is not None}

    # =========================================================================
    # OLLAMA-COMPATIBLE INVOKE METHODS
    # Override parent class to wrap OpenAI responses in Ollama-compatible format.
    # This enables code like `response.message.content` to work unchanged.
    # =========================================================================

    def invoke(
        self,
        messages: List[Message],
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> OllamaCompatibleResponse:
        """
        Send a chat request to llama.cpp server.

        Returns an Ollama-compatible response that supports:
            response.message.content  (Ollama-style)
            response.choices[0].message.content  (OpenAI-style)

        This enables code written for Ollama to work unchanged with LlamaCpp.

        Args:
            messages: List of Message objects
            response_format: Optional response format (dict or Pydantic model)
            tools: Optional list of tool definitions
            tool_choice: Optional tool choice specification

        Returns:
            OllamaCompatibleResponse: Response with .message.content access
        """
        # Call parent's invoke which returns OpenAI ChatCompletion
        openai_response = super().invoke(
            messages=messages,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
        )
        # Wrap in Ollama-compatible format
        return OllamaCompatibleResponse(openai_response)

    async def ainvoke(
        self,
        messages: List[Message],
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> OllamaCompatibleResponse:
        """
        Async version of invoke.

        Returns an Ollama-compatible response that supports:
            response.message.content  (Ollama-style)
            response.choices[0].message.content  (OpenAI-style)

        Args:
            messages: List of Message objects
            response_format: Optional response format (dict or Pydantic model)
            tools: Optional list of tool definitions
            tool_choice: Optional tool choice specification

        Returns:
            OllamaCompatibleResponse: Response with .message.content access
        """
        # Call parent's ainvoke which returns OpenAI ChatCompletion
        openai_response = await super().ainvoke(
            messages=messages,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
        )
        # Wrap in Ollama-compatible format
        return OllamaCompatibleResponse(openai_response)


@dataclass
class LlamaCppMultiModel:
    """
    Helper class for managing multiple LlamaCpp models on the same server.

    Provides convenient access to different models for different tasks.

    Example:
        models = LlamaCppMultiModel(
            base_url="http://localhost:8080/v1",
            utility_model="qwen2.5-7b",
            analysis_model="gpt-oss-20b",
        )

        # Use different models for different tasks
        utility_llm = models.utility  # For tool calling, JSON, fast tasks
        analysis_llm = models.analysis  # For summarization, long-form output
    """

    base_url: str = "http://localhost:8080/v1"
    utility_model: str = "qwen2.5-7b"
    analysis_model: str = "gpt-oss-20b"
    json_model: Optional[str] = None  # Defaults to utility_model

    # Shared settings
    temperature: float = 0.2
    cache_prompt: bool = True

    def __post_init__(self):
        if self.json_model is None:
            self.json_model = self.utility_model

    @property
    def utility(self) -> LlamaCpp:
        """Get LLM for utility tasks (tool calling, JSON, fast responses)."""
        return LlamaCpp(
            id=self.utility_model,
            base_url=self.base_url,
            temperature=self.temperature,
            cache_prompt=self.cache_prompt,
        )

    @property
    def analysis(self) -> LlamaCpp:
        """Get LLM for analysis tasks (summarization, long-form output)."""
        return LlamaCpp(
            id=self.analysis_model,
            base_url=self.base_url,
            temperature=self.temperature,
            cache_prompt=self.cache_prompt,
        )

    @property
    def json(self) -> LlamaCpp:
        """Get LLM optimized for JSON output."""
        return LlamaCpp(
            id=self.json_model,
            base_url=self.base_url,
            temperature=0.1,  # Low temperature for deterministic JSON
            cache_prompt=self.cache_prompt,
        )
