"""
Complete OpenCode Website Backend
FastAPI server with all routes for chat, sessions, messages, and tools
"""

import os
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from jose import JWTError, jwt
from dotenv import load_dotenv
import uuid
from enum import Enum
from sse_starlette.sse import EventSourceResponse

# Load environment variables
load_dotenv()

# Import our modules
from ai_providers import SUPPORTED_MODELS, get_model
from tools import get_all_tools, get_tool_schemas
from chat_handler import chat_handler

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("JWT_SECRET", "your-secret-key-change-this")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Database
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/opencode")

# Global variables
db = None
websocket_manager = None

# Pydantic Models (reusing from previous server.py)
class UserRegistration(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

class CreateSessionRequest(BaseModel):
    title: Optional[str] = "New Chat"

class SendMessageRequest(BaseModel):
    content: str
    session_id: str
    model: Optional[str] = "gpt-4o"
    stream: bool = True

class ConfigurationRequest(BaseModel):
    providers: Optional[Dict[str, Dict[str, Any]]] = None
    settings: Optional[Dict[str, Any]] = None

class FileOperationRequest(BaseModel):
    operation: str  # "read", "write", "delete", "list", "search"
    path: str
    content: Optional[str] = None
    pattern: Optional[str] = None

class CustomCommandRequest(BaseModel):
    name: str
    description: str
    content: str

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)

    async def send_json_message(self, message: dict, user_id: str):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_json(message)

# Database initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global db, websocket_manager
    try:
        client = AsyncIOMotorClient(MONGO_URL)
        db = client.get_database()
        websocket_manager = ConnectionManager()
        
        # Create indexes
        await create_indexes()
        
        logging.info("Database connected successfully")
    except Exception as e:
        logging.error(f"Failed to connect to database: {e}")
        raise
    
    yield
    
    # Shutdown
    if db:
        db.client.close()

# Create FastAPI app
app = FastAPI(
    title="OpenCode Website API",
    description="Complete backend API for OpenCode - Terminal-based AI Assistant",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set global db reference for other modules
import chat_handler
import server
server.db = db
server.websocket_manager = websocket_manager
chat_handler.db = db
chat_handler.websocket_manager = websocket_manager

async def create_indexes():
    """Create database indexes for performance"""
    try:
        # Users collection indexes
        await db.users.create_index("email", unique=True)
        
        # Sessions collection indexes
        await db.sessions.create_index("user_id")
        await db.sessions.create_index("created_at")
        
        # Messages collection indexes
        await db.messages.create_index("session_id")
        await db.messages.create_index("created_at")
        
        # Custom commands collection indexes
        await db.custom_commands.create_index("user_id")
        
        logging.info("Database indexes created successfully")
    except Exception as e:
        logging.error(f"Failed to create indexes: {e}")

# Utility functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = await db.users.find_one({"_id": user_id})
    if user is None:
        raise credentials_exception
    
    return User(
        id=user["_id"],
        email=user["email"],
        full_name=user.get("full_name"),
        created_at=user["created_at"],
        updated_at=user["updated_at"]
    )

# ============== AUTHENTICATION ROUTES ==============

@app.post("/api/auth/register", response_model=Token)
async def register(user_data: UserRegistration):
    # Check if user exists
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    user_id = str(uuid.uuid4())
    hashed_password = get_password_hash(user_data.password)
    now = datetime.utcnow()
    
    user_doc = {
        "_id": user_id,
        "email": user_data.email,
        "full_name": user_data.full_name,
        "hashed_password": hashed_password,
        "created_at": now,
        "updated_at": now
    }
    
    await db.users.insert_one(user_doc)
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_id}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/auth/login", response_model=Token)
async def login(user_data: UserLogin):
    user = await db.users.find_one({"email": user_data.email})
    if not user or not verify_password(user_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["_id"]}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/auth/me", response_model=User)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user

# ============== HEALTH CHECK ==============

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# ============== SESSION ROUTES ==============

@app.post("/api/sessions")
async def create_session(
    request: CreateSessionRequest,
    current_user: User = Depends(get_current_user)
):
    session_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    session_doc = {
        "_id": session_id,
        "user_id": current_user.id,
        "title": request.title or "New Chat",
        "message_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost": 0.0,
        "summary_message_id": None,
        "parent_session_id": None,
        "created_at": now,
        "updated_at": now
    }
    
    await db.sessions.insert_one(session_doc)
    
    return {
        "id": session_id,
        "user_id": current_user.id,
        "title": session_doc["title"],
        "message_count": session_doc["message_count"],
        "prompt_tokens": session_doc["prompt_tokens"],
        "completion_tokens": session_doc["completion_tokens"],
        "cost": session_doc["cost"],
        "created_at": session_doc["created_at"],
        "updated_at": session_doc["updated_at"]
    }

@app.get("/api/sessions")
async def list_sessions(current_user: User = Depends(get_current_user)):
    sessions = await db.sessions.find(
        {"user_id": current_user.id}
    ).sort("updated_at", -1).to_list(100)
    
    return [
        {
            "id": session["_id"],
            "user_id": session["user_id"],
            "title": session["title"],
            "message_count": session["message_count"],
            "prompt_tokens": session["prompt_tokens"],
            "completion_tokens": session["completion_tokens"],
            "cost": session["cost"],
            "summary_message_id": session.get("summary_message_id"),
            "parent_session_id": session.get("parent_session_id"),
            "created_at": session["created_at"],
            "updated_at": session["updated_at"]
        }
        for session in sessions
    ]

@app.get("/api/sessions/{session_id}")
async def get_session(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    session = await db.sessions.find_one({
        "_id": session_id,
        "user_id": current_user.id
    })
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "id": session["_id"],
        "user_id": session["user_id"],
        "title": session["title"],
        "message_count": session["message_count"],
        "prompt_tokens": session["prompt_tokens"],
        "completion_tokens": session["completion_tokens"],
        "cost": session["cost"],
        "summary_message_id": session.get("summary_message_id"),
        "parent_session_id": session.get("parent_session_id"),
        "created_at": session["created_at"],
        "updated_at": session["updated_at"]
    }

@app.delete("/api/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    # Check if session exists and belongs to user
    session = await db.sessions.find_one({
        "_id": session_id,
        "user_id": current_user.id
    })
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Delete session and its messages
    await db.sessions.delete_one({"_id": session_id})
    await db.messages.delete_many({"session_id": session_id})
    
    return {"message": "Session deleted successfully"}

# ============== MESSAGE ROUTES ==============

@app.get("/api/sessions/{session_id}/messages")
async def get_messages(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    # Verify session belongs to user
    session = await db.sessions.find_one({
        "_id": session_id,
        "user_id": current_user.id
    })
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = await db.messages.find(
        {"session_id": session_id}
    ).sort("created_at", 1).to_list(None)
    
    return [
        {
            "id": msg["_id"],
            "session_id": msg["session_id"],
            "role": msg["role"],
            "content": "".join([
                part["data"]["text"] for part in msg.get("parts", [])
                if part["type"] == "text"
            ]),
            "parts": msg.get("parts", []),
            "model": msg.get("model"),
            "created_at": msg["created_at"],
            "updated_at": msg["updated_at"],
            "finished_at": msg.get("finished_at")
        }
        for msg in messages
    ]

# ============== CHAT ROUTES ==============

@app.post("/api/chat/send")
async def send_message_stream(
    request: SendMessageRequest,
    current_user: User = Depends(get_current_user)
):
    """Send a message and get streaming response"""
    
    async def generate():
        async for event in chat_handler.send_message(
            user=current_user,
            session_id=request.session_id,
            content=request.content,
            model_id=request.model or "gpt-4o",
            stream=request.stream
        ):
            yield f"data: {json.dumps(event)}\n\n"
    
    return EventSourceResponse(generate())

@app.post("/api/chat/send-sync")
async def send_message_sync(
    request: SendMessageRequest,
    current_user: User = Depends(get_current_user)
):
    """Send a message and get non-streaming response"""
    
    events = []
    async for event in chat_handler.send_message(
        user=current_user,
        session_id=request.session_id,
        content=request.content,
        model_id=request.model or "gpt-4o",
        stream=False
    ):
        events.append(event)
    
    return {"events": events}

# ============== MODEL ROUTES ==============

@app.get("/api/models")
async def get_models():
    """Get all available AI models"""
    models = []
    for model_id, model in SUPPORTED_MODELS.items():
        models.append({
            "id": model.id,
            "name": model.name,
            "provider": model.provider.value,
            "cost_per_1m_in": model.cost_per_1m_in,
            "cost_per_1m_out": model.cost_per_1m_out,
            "context_window": model.context_window,
            "can_reason": model.can_reason,
            "supports_attachments": model.supports_attachments
        })
    
    return {"models": models}

@app.get("/api/models/{model_id}")
async def get_model_info(model_id: str):
    """Get information about a specific model"""
    model = get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {
        "id": model.id,
        "name": model.name,
        "provider": model.provider.value,
        "cost_per_1m_in": model.cost_per_1m_in,
        "cost_per_1m_out": model.cost_per_1m_out,
        "context_window": model.context_window,
        "can_reason": model.can_reason,
        "supports_attachments": model.supports_attachments
    }

# ============== TOOLS ROUTES ==============

@app.get("/api/tools")
async def get_tools():
    """Get all available tools"""
    tools = get_all_tools()
    return {
        "tools": [
            {
                "name": tool.info().name,
                "description": tool.info().description,
                "parameters": tool.info().parameters,
                "required": tool.info().required
            }
            for tool in tools
        ]
    }

@app.get("/api/tools/schemas")
async def get_tool_schemas_endpoint():
    """Get OpenAI-compatible tool schemas"""
    return {"schemas": get_tool_schemas()}

# ============== FILE OPERATIONS ==============

@app.post("/api/files/operation")
async def file_operation(
    request: FileOperationRequest,
    current_user: User = Depends(get_current_user)
):
    """Perform file operations"""
    
    from tools import get_tool, ToolCall as ToolCallModel
    
    # Map operations to tools
    operation_tool_map = {
        "read": "view",
        "write": "write", 
        "search": "grep",
        "list": "glob"
    }
    
    tool_name = operation_tool_map.get(request.operation)
    if not tool_name:
        raise HTTPException(status_code=400, detail="Unsupported operation")
    
    tool = get_tool(tool_name)
    if not tool:
        raise HTTPException(status_code=500, detail="Tool not available")
    
    # Prepare tool parameters
    params = {"file_path": request.path} if request.operation == "read" else {"path": request.path}
    
    if request.content:
        params["content"] = request.content
    if request.pattern:
        params["pattern"] = request.pattern
    
    # Execute tool
    tool_call = ToolCallModel(
        id=str(uuid.uuid4()),
        name=tool_name,
        input=json.dumps(params)
    )
    
    result = await tool.run(tool_call)
    
    return {
        "success": not result.is_error,
        "content": result.content,
        "metadata": result.metadata
    }

# ============== WEBSOCKET ==============

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket_manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle websocket messages if needed
            await websocket_manager.send_personal_message(f"Echo: {data}", user_id)
    except WebSocketDisconnect:
        websocket_manager.disconnect(user_id)

# ============== CONFIGURATION ==============

@app.get("/api/config")
async def get_configuration(current_user: User = Depends(get_current_user)):
    """Get user configuration"""
    config = await db.configurations.find_one({"user_id": current_user.id})
    
    if not config:
        # Return default configuration
        return {
            "providers": {},
            "settings": {
                "theme": "opencode",
                "auto_compact": True,
                "default_model": "gpt-4o"
            }
        }
    
    return {
        "providers": config.get("providers", {}),
        "settings": config.get("settings", {})
    }

@app.post("/api/config")
async def update_configuration(
    request: ConfigurationRequest,
    current_user: User = Depends(get_current_user)
):
    """Update user configuration"""
    
    update_data = {}
    if request.providers:
        update_data["providers"] = request.providers
    if request.settings:
        update_data["settings"] = request.settings
    
    update_data["updated_at"] = datetime.utcnow()
    
    await db.configurations.update_one(
        {"user_id": current_user.id},
        {"$set": update_data},
        upsert=True
    )
    
    return {"message": "Configuration updated successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)