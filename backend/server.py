"""
OpenCode Website Backend
FastAPI server that replicates all functionality from the OpenCode CLI
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from jose import JWTError, jwt
from dotenv import load_dotenv
import json
import uuid
from enum import Enum

# Load environment variables
load_dotenv()

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

# Pydantic Models
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

class FinishReason(str, Enum):
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    TOOL_USE = "tool_use"
    CANCELED = "canceled"
    ERROR = "error"
    PERMISSION_DENIED = "permission_denied"

class AIProvider(str, Enum):
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

class ContentPart(BaseModel):
    type: str
    data: Dict[str, Any]

class Message(BaseModel):
    id: str
    session_id: str
    role: MessageRole
    parts: List[ContentPart]
    model: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    finished_at: Optional[datetime] = None

class Session(BaseModel):
    id: str
    user_id: str
    title: str
    message_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    summary_message_id: Optional[str] = None
    parent_session_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class CreateSessionRequest(BaseModel):
    title: Optional[str] = "New Chat"

class SendMessageRequest(BaseModel):
    content: str
    session_id: str
    model: Optional[str] = None
    stream: bool = True

class ToolCall(BaseModel):
    id: str
    name: str
    input: str
    type: str = "function"

class ToolResult(BaseModel):
    tool_call_id: str
    content: str
    is_error: bool = False
    metadata: Optional[str] = None

class FileOperation(BaseModel):
    operation: str  # "read", "write", "delete", "list", "search"
    path: str
    content: Optional[str] = None
    pattern: Optional[str] = None

class CustomCommand(BaseModel):
    id: str
    name: str
    description: str
    content: str
    created_at: datetime

class Configuration(BaseModel):
    providers: Dict[str, Dict[str, Any]]
    models: Dict[str, Dict[str, Any]]
    themes: Dict[str, Any]
    settings: Dict[str, Any]

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
    description="Backend API for OpenCode - Terminal-based AI Assistant converted to web",
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

# Authentication routes
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

# Health check
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Sessions routes
@app.post("/api/sessions", response_model=Session)
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
    
    return Session(
        id=session_id,
        user_id=current_user.id,
        title=session_doc["title"],
        message_count=session_doc["message_count"],
        prompt_tokens=session_doc["prompt_tokens"],
        completion_tokens=session_doc["completion_tokens"],
        cost=session_doc["cost"],
        summary_message_id=session_doc["summary_message_id"],
        parent_session_id=session_doc["parent_session_id"],
        created_at=session_doc["created_at"],
        updated_at=session_doc["updated_at"]
    )

@app.get("/api/sessions", response_model=List[Session])
async def list_sessions(current_user: User = Depends(get_current_user)):
    sessions = await db.sessions.find(
        {"user_id": current_user.id}
    ).sort("updated_at", -1).to_list(100)
    
    return [
        Session(
            id=session["_id"],
            user_id=session["user_id"],
            title=session["title"],
            message_count=session["message_count"],
            prompt_tokens=session["prompt_tokens"],
            completion_tokens=session["completion_tokens"],
            cost=session["cost"],
            summary_message_id=session.get("summary_message_id"),
            parent_session_id=session.get("parent_session_id"),
            created_at=session["created_at"],
            updated_at=session["updated_at"]
        )
        for session in sessions
    ]

@app.get("/api/sessions/{session_id}", response_model=Session)
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
    
    return Session(
        id=session["_id"],
        user_id=session["user_id"],
        title=session["title"],
        message_count=session["message_count"],
        prompt_tokens=session["prompt_tokens"],
        completion_tokens=session["completion_tokens"],
        cost=session["cost"],
        summary_message_id=session.get("summary_message_id"),
        parent_session_id=session.get("parent_session_id"),
        created_at=session["created_at"],
        updated_at=session["updated_at"]
    )

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)