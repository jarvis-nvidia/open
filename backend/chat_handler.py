"""
Chat Handler for OpenCode
Manages AI conversations, streaming responses, and tool execution
"""

import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
import uuid

from ai_providers import create_provider, get_model, SUPPORTED_MODELS, ProviderEvent
from tools import get_tool, get_tool_schemas, ToolResponse
from server import db, websocket_manager, User, Message, Session, ContentPart, MessageRole

class ChatHandler:
    """Handles AI chat interactions with tool support"""
    
    def __init__(self):
        self.active_sessions = {}  # session_id -> cancel_function
    
    async def send_message(
        self, 
        user: User,
        session_id: str, 
        content: str, 
        model_id: str = "gpt-4o",
        stream: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Send a message and get AI response"""
        
        try:
            # Get session
            session = await db.sessions.find_one({
                "_id": session_id,
                "user_id": user.id
            })
            
            if not session:
                yield {
                    "type": "error",
                    "content": "Session not found"
                }
                return
            
            # Get model
            model = get_model(model_id)
            if not model:
                yield {
                    "type": "error", 
                    "content": f"Model not supported: {model_id}"
                }
                return
            
            # Get API key for provider
            api_key = self._get_api_key(model.provider.value)
            if not api_key:
                yield {
                    "type": "error",
                    "content": f"API key not configured for provider: {model.provider.value}"
                }
                return
            
            # Create user message
            user_message = await self._create_message(
                session_id=session_id,
                role=MessageRole.USER,
                content=content
            )
            
            yield {
                "type": "message",
                "data": {
                    "id": user_message["_id"],
                    "role": "user",
                    "content": content,
                    "created_at": user_message["created_at"].isoformat()
                }
            }
            
            # Create assistant message  
            assistant_message = await self._create_message(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content="",
                model=model_id
            )
            
            # Get conversation history
            messages = await self._get_conversation_history(session_id)
            
            # Create AI provider
            provider = create_provider(model.provider.value, model_id, api_key)
            
            # Get available tools
            tools = get_tool_schemas()
            
            # Generate system message
            system_message = self._get_system_message()
            
            if stream:
                # Stream response
                async for event in self._stream_ai_response(
                    provider=provider,
                    messages=messages,
                    tools=tools,
                    system_message=system_message,
                    assistant_message_id=assistant_message["_id"],
                    session_id=session_id,
                    user=user
                ):
                    yield event
            else:
                # Non-streaming response
                response = await provider.send_message(
                    messages=messages,
                    tools=tools,
                    system_message=system_message
                )
                
                # Update assistant message
                await db.messages.update_one(
                    {"_id": assistant_message["_id"]},
                    {
                        "$set": {
                            "parts": [{"type": "text", "data": {"text": response.content}}],
                            "updated_at": datetime.utcnow(),
                            "finished_at": datetime.utcnow()
                        }
                    }
                )
                
                yield {
                    "type": "message",
                    "data": {
                        "id": assistant_message["_id"],
                        "role": "assistant",
                        "content": response.content,
                        "finished": True
                    }
                }
                
                # Update session stats
                await self._update_session_stats(session_id, response.usage)
                
        except Exception as e:
            logging.error(f"Chat error: {str(e)}")
            yield {
                "type": "error",
                "content": f"Chat error: {str(e)}"
            }
    
    async def _stream_ai_response(
        self,
        provider,
        messages: List[Dict[str, Any]], 
        tools: List[Dict[str, Any]],
        system_message: str,
        assistant_message_id: str,
        session_id: str,
        user: User
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream AI response with tool support"""
        
        current_content = ""
        current_thinking = ""
        tool_calls = []
        
        try:
            async for event in provider.stream_response(
                messages=messages,
                tools=tools,
                system_message=system_message
            ):
                if event.type == "content_delta":
                    current_content += event.content
                    
                    # Update message in database
                    await db.messages.update_one(
                        {"_id": assistant_message_id},
                        {
                            "$set": {
                                "parts": [{"type": "text", "data": {"text": current_content}}],
                                "updated_at": datetime.utcnow()
                            }
                        }
                    )
                    
                    # Send to client
                    yield {
                        "type": "content_delta",
                        "content": event.content,
                        "message_id": assistant_message_id
                    }
                    
                    # Send via WebSocket if connected
                    if websocket_manager:
                        await websocket_manager.send_json_message({
                            "type": "content_delta",
                            "content": event.content,
                            "message_id": assistant_message_id,
                            "session_id": session_id
                        }, user.id)
                
                elif event.type == "thinking_delta":
                    current_thinking += event.thinking
                    
                    yield {
                        "type": "thinking_delta", 
                        "content": event.thinking,
                        "message_id": assistant_message_id
                    }
                
                elif event.type == "tool_use_start":
                    tool_calls.append(event.tool_call)
                    
                    yield {
                        "type": "tool_use_start",
                        "tool_call": event.tool_call,
                        "message_id": assistant_message_id
                    }
                
                elif event.type == "complete":
                    # Handle tool calls if any
                    if tool_calls:
                        tool_results = await self._execute_tools(tool_calls, session_id, user)
                        
                        # Add tool results to conversation
                        for result in tool_results:
                            await self._create_message(
                                session_id=session_id,
                                role=MessageRole.TOOL,
                                content=result.content,
                                tool_call_id=result.tool_call_id if hasattr(result, 'tool_call_id') else None
                            )
                            
                            yield {
                                "type": "tool_result",
                                "result": result.dict()
                            }
                        
                        # Continue conversation with tool results
                        updated_messages = await self._get_conversation_history(session_id)
                        
                        async for continue_event in provider.stream_response(
                            messages=updated_messages,
                            tools=tools,
                            system_message=system_message
                        ):
                            # Handle continuation response (recursive)
                            if continue_event.type == "content_delta":
                                current_content += continue_event.content
                                
                                await db.messages.update_one(
                                    {"_id": assistant_message_id},
                                    {
                                        "$set": {
                                            "parts": [{"type": "text", "data": {"text": current_content}}],
                                            "updated_at": datetime.utcnow()
                                        }
                                    }
                                )
                                
                                yield {
                                    "type": "content_delta",
                                    "content": continue_event.content,
                                    "message_id": assistant_message_id
                                }
                            
                            elif continue_event.type == "complete":
                                break
                    
                    # Finalize message
                    await db.messages.update_one(
                        {"_id": assistant_message_id},
                        {
                            "$set": {
                                "finished_at": datetime.utcnow(),
                                "updated_at": datetime.utcnow()
                            }
                        }
                    )
                    
                    # Update session stats
                    if event.response:
                        await self._update_session_stats(session_id, event.response.usage)
                    
                    yield {
                        "type": "complete",
                        "message_id": assistant_message_id,
                        "session_id": session_id
                    }
                    
                elif event.type == "error":
                    yield {
                        "type": "error",
                        "content": event.error
                    }
                    break
                    
        except Exception as e:
            logging.error(f"Streaming error: {str(e)}")
            yield {
                "type": "error",
                "content": f"Streaming error: {str(e)}"
            }
    
    async def _execute_tools(
        self, 
        tool_calls: List[Dict[str, Any]], 
        session_id: str, 
        user: User
    ) -> List[Dict[str, Any]]:
        """Execute tool calls"""
        results = []
        
        for tool_call in tool_calls:
            try:
                tool_name = tool_call["name"]
                tool = get_tool(tool_name)
                
                if not tool:
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "content": f"Tool not found: {tool_name}",
                        "is_error": True
                    })
                    continue
                
                # Execute tool
                from tools import ToolCall as ToolCallModel
                tool_call_obj = ToolCallModel(
                    id=tool_call["id"],
                    name=tool_name,
                    input=tool_call["input"]
                )
                
                result = await tool.run(tool_call_obj)
                
                results.append({
                    "tool_call_id": tool_call["id"],
                    "content": result.content,
                    "is_error": result.is_error,
                    "metadata": result.metadata
                })
                
            except Exception as e:
                logging.error(f"Tool execution error: {str(e)}")
                results.append({
                    "tool_call_id": tool_call.get("id", "unknown"),
                    "content": f"Tool execution error: {str(e)}",
                    "is_error": True
                })
        
        return results
    
    async def _create_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        model: Optional[str] = None,
        tool_call_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new message in the database"""
        
        message_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        parts = [{"type": "text", "data": {"text": content}}]
        
        message_doc = {
            "_id": message_id,
            "session_id": session_id,
            "role": role.value,
            "parts": parts,
            "model": model,
            "tool_call_id": tool_call_id,
            "created_at": now,
            "updated_at": now,
            "finished_at": now if role != MessageRole.ASSISTANT else None
        }
        
        await db.messages.insert_one(message_doc)
        
        # Update session message count
        await db.sessions.update_one(
            {"_id": session_id},
            {
                "$inc": {"message_count": 1},
                "$set": {"updated_at": now}
            }
        )
        
        return message_doc
    
    async def _get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history formatted for AI provider"""
        
        messages = await db.messages.find(
            {"session_id": session_id}
        ).sort("created_at", 1).to_list(None)
        
        formatted_messages = []
        
        for msg in messages:
            # Extract text content from parts
            content = ""
            for part in msg.get("parts", []):
                if part["type"] == "text":
                    content += part["data"]["text"]
            
            if content.strip():  # Only include messages with content
                formatted_messages.append({
                    "role": msg["role"],
                    "content": content
                })
        
        return formatted_messages
    
    async def _update_session_stats(self, session_id: str, usage):
        """Update session token usage and cost statistics"""
        
        try:
            session = await db.sessions.find_one({"_id": session_id})
            if not session:
                return
            
            # Calculate cost (this is simplified - real implementation would use model pricing)
            cost_per_input_token = 0.00001  # $0.01 per 1K tokens
            cost_per_output_token = 0.00003  # $0.03 per 1K tokens
            
            additional_cost = (
                usage.input_tokens * cost_per_input_token +
                usage.output_tokens * cost_per_output_token
            )
            
            await db.sessions.update_one(
                {"_id": session_id},
                {
                    "$inc": {
                        "prompt_tokens": usage.input_tokens,
                        "completion_tokens": usage.output_tokens,
                        "cost": additional_cost
                    },
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
            
        except Exception as e:
            logging.error(f"Error updating session stats: {str(e)}")
    
    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider"""
        import os
        
        key_mapping = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY", 
            "gemini": "GEMINI_API_KEY",
            "groq": "GROQ_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "xai": "XAI_API_KEY",
            "azure": "AZURE_OPENAI_API_KEY",
            "copilot": "GITHUB_TOKEN"
        }
        
        env_var = key_mapping.get(provider)
        return os.getenv(env_var) if env_var else None
    
    def _get_system_message(self) -> str:
        """Get system message for AI assistant"""
        return """You are OpenCode, a powerful AI assistant designed to help developers with coding tasks.

You have access to various tools that allow you to:
- View and edit files
- Search for files and content 
- Execute bash commands
- Fetch data from URLs
- And more

Always be helpful, accurate, and explain your reasoning. When using tools, explain what you're doing and why.

If you need to perform file operations or run commands, use the appropriate tools. Always confirm destructive actions before proceeding.

Provide clear, well-structured responses with code examples when relevant."""

# Global chat handler instance
chat_handler = ChatHandler()