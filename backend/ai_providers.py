"""
AI Provider integrations for OpenCode
Supports OpenAI, Anthropic, Gemini, Groq, and other providers
"""

import os
import json
import httpx
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from enum import Enum
from pydantic import BaseModel
import openai
from anthropic import Anthropic
import google.generativeai as genai
from groq import Groq

class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic" 
    GEMINI = "gemini"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    XAI = "xai"
    AZURE = "azure"
    BEDROCK = "bedrock"
    VERTEXAI = "vertexai"
    COPILOT = "copilot"
    LOCAL = "local"

class AIModel(BaseModel):
    id: str
    name: str
    provider: ModelProvider
    api_model: str
    cost_per_1m_in: float
    cost_per_1m_out: float
    context_window: int
    default_max_tokens: int
    can_reason: bool = False
    supports_attachments: bool = False

class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

class ProviderResponse(BaseModel):
    content: str
    tool_calls: List[Dict[str, Any]] = []
    usage: TokenUsage
    finish_reason: str

class ProviderEvent(BaseModel):
    type: str  # "content_start", "content_delta", "tool_use_start", etc.
    content: Optional[str] = None
    thinking: Optional[str] = None
    tool_call: Optional[Dict[str, Any]] = None
    response: Optional[ProviderResponse] = None
    error: Optional[str] = None

# Supported Models Configuration
SUPPORTED_MODELS = {
    # OpenAI Models
    "gpt-4.1": AIModel(
        id="gpt-4.1",
        name="GPT-4.1",
        provider=ModelProvider.OPENAI,
        api_model="gpt-4.1",
        cost_per_1m_in=5.0,
        cost_per_1m_out=15.0,
        context_window=128000,
        default_max_tokens=4096,
        can_reason=True,
        supports_attachments=True
    ),
    "gpt-4.1-mini": AIModel(
        id="gpt-4.1-mini",
        name="GPT-4.1 Mini",
        provider=ModelProvider.OPENAI,
        api_model="gpt-4.1-mini",
        cost_per_1m_in=0.15,
        cost_per_1m_out=0.6,
        context_window=128000,
        default_max_tokens=4096,
        can_reason=True,
        supports_attachments=True
    ),
    "gpt-4o": AIModel(
        id="gpt-4o",
        name="GPT-4o",
        provider=ModelProvider.OPENAI,
        api_model="gpt-4o",
        cost_per_1m_in=2.5,
        cost_per_1m_out=10.0,
        context_window=128000,
        default_max_tokens=4096,
        supports_attachments=True
    ),
    "gpt-4o-mini": AIModel(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        provider=ModelProvider.OPENAI,
        api_model="gpt-4o-mini",
        cost_per_1m_in=0.15,
        cost_per_1m_out=0.6,
        context_window=128000,
        default_max_tokens=4096,
        supports_attachments=True
    ),
    "o1": AIModel(
        id="o1",
        name="o1",
        provider=ModelProvider.OPENAI,
        api_model="o1",
        cost_per_1m_in=15.0,
        cost_per_1m_out=60.0,
        context_window=200000,
        default_max_tokens=100000,
        can_reason=True
    ),
    "o1-mini": AIModel(
        id="o1-mini",
        name="o1-mini",
        provider=ModelProvider.OPENAI,
        api_model="o1-mini",
        cost_per_1m_in=3.0,
        cost_per_1m_out=12.0,
        context_window=128000,
        default_max_tokens=65536,
        can_reason=True
    ),
    
    # Anthropic Models
    "claude-4-sonnet": AIModel(
        id="claude-4-sonnet",
        name="Claude 4 Sonnet",
        provider=ModelProvider.ANTHROPIC,
        api_model="claude-4-sonnet-20250110",
        cost_per_1m_in=3.0,
        cost_per_1m_out=15.0,
        context_window=200000,
        default_max_tokens=8192,
        can_reason=True,
        supports_attachments=True
    ),
    "claude-3.7-sonnet": AIModel(
        id="claude-3.7-sonnet",
        name="Claude 3.7 Sonnet",
        provider=ModelProvider.ANTHROPIC,
        api_model="claude-3-7-sonnet-20250115",
        cost_per_1m_in=3.0,
        cost_per_1m_out=15.0,
        context_window=200000,
        default_max_tokens=8192,
        can_reason=True,
        supports_attachments=True
    ),
    "claude-3.5-sonnet": AIModel(
        id="claude-3.5-sonnet",
        name="Claude 3.5 Sonnet",
        provider=ModelProvider.ANTHROPIC,
        api_model="claude-3-5-sonnet-20241022",
        cost_per_1m_in=3.0,
        cost_per_1m_out=15.0,
        context_window=200000,
        default_max_tokens=8192,
        supports_attachments=True
    ),
    "claude-3.5-haiku": AIModel(
        id="claude-3.5-haiku",
        name="Claude 3.5 Haiku",
        provider=ModelProvider.ANTHROPIC,
        api_model="claude-3-5-haiku-20241022",
        cost_per_1m_in=0.8,
        cost_per_1m_out=4.0,
        context_window=200000,
        default_max_tokens=8192,
        supports_attachments=True
    ),
    
    # Google Gemini Models
    "gemini-2.5": AIModel(
        id="gemini-2.5",
        name="Gemini 2.5 Pro",
        provider=ModelProvider.GEMINI,
        api_model="gemini-2.5-pro-exp-03-25",
        cost_per_1m_in=0.0,
        cost_per_1m_out=0.0,
        context_window=2000000,
        default_max_tokens=8192,
        supports_attachments=True
    ),
    "gemini-2.0-flash": AIModel(
        id="gemini-2.0-flash",
        name="Gemini 2.0 Flash",
        provider=ModelProvider.GEMINI,
        api_model="gemini-2.0-flash",
        cost_per_1m_in=0.075,
        cost_per_1m_out=0.3,
        context_window=1000000,
        default_max_tokens=8192,
        supports_attachments=True
    ),
    
    # Groq Models
    "llama-4-maverick": AIModel(
        id="llama-4-maverick",
        name="Llama 4 Maverick",
        provider=ModelProvider.GROQ,
        api_model="llama-4-maverick-17b-128e-instruct",
        cost_per_1m_in=0.0,
        cost_per_1m_out=0.0,
        context_window=128000,
        default_max_tokens=4096
    ),
    "qwen-qwq": AIModel(
        id="qwen-qwq",
        name="QWEN QWQ-32b",
        provider=ModelProvider.GROQ,
        api_model="qwen-qwq-32b-preview",
        cost_per_1m_in=0.0,
        cost_per_1m_out=0.0,
        context_window=32768,
        default_max_tokens=4096,
        can_reason=True
    )
}

class AIProviderBase:
    """Base class for AI providers"""
    
    def __init__(self, api_key: str, model: AIModel):
        self.api_key = api_key
        self.model = model
    
    async def send_message(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None
    ) -> ProviderResponse:
        """Send a message and get a response"""
        raise NotImplementedError
    
    async def stream_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None
    ) -> AsyncGenerator[ProviderEvent, None]:
        """Stream a response"""
        raise NotImplementedError

class OpenAIProvider(AIProviderBase):
    """OpenAI provider implementation"""
    
    def __init__(self, api_key: str, model: AIModel):
        super().__init__(api_key, model)
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    async def send_message(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None
    ) -> ProviderResponse:
        try:
            # Prepare messages
            formatted_messages = self._format_messages(messages, system_message)
            
            # Prepare request parameters
            params = {
                "model": self.model.api_model,
                "messages": formatted_messages,
                "max_tokens": max_tokens or self.model.default_max_tokens,
                "temperature": 0.7
            }
            
            if tools:
                params["tools"] = tools
                params["tool_choice"] = "auto"
            
            # Add reasoning effort for reasoning models
            if self.model.can_reason:
                params["reasoning_effort"] = "medium"
            
            response = await self.client.chat.completions.create(**params)
            
            # Extract response data
            message = response.choices[0].message
            content = message.content or ""
            
            tool_calls = []
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "input": tool_call.function.arguments,
                        "type": "function"
                    })
            
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )
            
            return ProviderResponse(
                content=content,
                tool_calls=tool_calls,
                usage=usage,
                finish_reason=response.choices[0].finish_reason
            )
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def stream_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None
    ) -> AsyncGenerator[ProviderEvent, None]:
        try:
            # Prepare messages
            formatted_messages = self._format_messages(messages, system_message)
            
            # Prepare request parameters
            params = {
                "model": self.model.api_model,
                "messages": formatted_messages,
                "max_tokens": max_tokens or self.model.default_max_tokens,
                "temperature": 0.7,
                "stream": True
            }
            
            if tools:
                params["tools"] = tools
                params["tool_choice"] = "auto"
            
            # Add reasoning effort for reasoning models
            if self.model.can_reason:
                params["reasoning_effort"] = "medium"
            
            stream = await self.client.chat.completions.create(**params)
            
            async for chunk in stream:
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    if delta.content:
                        yield ProviderEvent(
                            type="content_delta",
                            content=delta.content
                        )
                    
                    if delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            if tool_call.function:
                                yield ProviderEvent(
                                    type="tool_use_start" if tool_call.function.name else "tool_use_delta",
                                    tool_call={
                                        "id": tool_call.id,
                                        "name": tool_call.function.name,
                                        "input": tool_call.function.arguments
                                    }
                                )
                    
                    if choice.finish_reason:
                        usage = TokenUsage()
                        if hasattr(chunk, 'usage') and chunk.usage:
                            usage = TokenUsage(
                                input_tokens=chunk.usage.prompt_tokens,
                                output_tokens=chunk.usage.completion_tokens
                            )
                        
                        yield ProviderEvent(
                            type="complete",
                            response=ProviderResponse(
                                content="",
                                tool_calls=[],
                                usage=usage,
                                finish_reason=choice.finish_reason
                            )
                        )
                        
        except Exception as e:
            yield ProviderEvent(
                type="error",
                error=str(e)
            )
    
    def _format_messages(self, messages: List[Dict[str, Any]], system_message: Optional[str] = None) -> List[Dict[str, Any]]:
        """Format messages for OpenAI API"""
        formatted = []
        
        if system_message:
            formatted.append({
                "role": "system",
                "content": system_message
            })
        
        for msg in messages:
            formatted.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        return formatted

class AnthropicProvider(AIProviderBase):
    """Anthropic Claude provider implementation"""
    
    def __init__(self, api_key: str, model: AIModel):
        super().__init__(api_key, model)
        self.client = Anthropic(api_key=api_key)
    
    async def send_message(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None
    ) -> ProviderResponse:
        try:
            # Format messages for Anthropic
            formatted_messages = self._format_messages(messages)
            
            params = {
                "model": self.model.api_model,
                "messages": formatted_messages,
                "max_tokens": max_tokens or self.model.default_max_tokens,
                "temperature": 0.7
            }
            
            if system_message:
                params["system"] = system_message
            
            if tools:
                params["tools"] = self._format_tools(tools)
            
            response = await asyncio.to_thread(
                self.client.messages.create,
                **params
            )
            
            # Extract content
            content = ""
            tool_calls = []
            
            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "input": json.dumps(block.input),
                        "type": "function"
                    })
            
            usage = TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens
            )
            
            return ProviderResponse(
                content=content,
                tool_calls=tool_calls,
                usage=usage,
                finish_reason=response.stop_reason
            )
            
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
    
    async def stream_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None
    ) -> AsyncGenerator[ProviderEvent, None]:
        try:
            # Format messages for Anthropic
            formatted_messages = self._format_messages(messages)
            
            params = {
                "model": self.model.api_model,
                "messages": formatted_messages,
                "max_tokens": max_tokens or self.model.default_max_tokens,
                "temperature": 0.7,
                "stream": True
            }
            
            if system_message:
                params["system"] = system_message
            
            if tools:
                params["tools"] = self._format_tools(tools)
            
            async def stream_generator():
                with self.client.messages.stream(**params) as stream:
                    for event in stream:
                        yield event
            
            async for event in stream_generator():
                if event.type == "content_block_delta":
                    if hasattr(event.delta, 'text'):
                        yield ProviderEvent(
                            type="content_delta",
                            content=event.delta.text
                        )
                elif event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        yield ProviderEvent(
                            type="tool_use_start",
                            tool_call={
                                "id": event.content_block.id,
                                "name": event.content_block.name,
                                "input": ""
                            }
                        )
                elif event.type == "message_stop":
                    yield ProviderEvent(
                        type="complete",
                        response=ProviderResponse(
                            content="",
                            tool_calls=[],
                            usage=TokenUsage(),
                            finish_reason="end_turn"
                        )
                    )
                        
        except Exception as e:
            yield ProviderEvent(
                type="error",
                error=str(e)
            )
    
    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for Anthropic API"""
        formatted = []
        for msg in messages:
            formatted.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        return formatted
    
    def _format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools for Anthropic API"""
        formatted = []
        for tool in tools:
            formatted.append({
                "name": tool["function"]["name"],
                "description": tool["function"]["description"],
                "input_schema": tool["function"]["parameters"]
            })
        return formatted

# Provider factory
def create_provider(provider_name: str, model_id: str, api_key: str) -> AIProviderBase:
    """Create an AI provider instance"""
    if model_id not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_id}")
    
    model = SUPPORTED_MODELS[model_id]
    
    if provider_name == ModelProvider.OPENAI:
        return OpenAIProvider(api_key, model)
    elif provider_name == ModelProvider.ANTHROPIC:
        return AnthropicProvider(api_key, model)
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")

# Get available models for a provider
def get_provider_models(provider_name: str) -> List[AIModel]:
    """Get all available models for a provider"""
    return [model for model in SUPPORTED_MODELS.values() if model.provider == provider_name]

# Get model by ID
def get_model(model_id: str) -> Optional[AIModel]:
    """Get a model by its ID"""
    return SUPPORTED_MODELS.get(model_id)