"""
Tools system for OpenCode - equivalent to the CLI tools
Includes file operations, bash execution, search, and other utilities
"""

import os
import subprocess
import json
import re
import fnmatch
import aiofiles
import asyncio
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from pydantic import BaseModel
import httpx
import urllib.parse

class ToolInfo(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str]

class ToolResponse(BaseModel):
    type: str = "text"  # "text" or "image"
    content: str
    metadata: Optional[str] = None
    is_error: bool = False

class ToolCall(BaseModel):
    id: str
    name: str
    input: str

class BaseTool:
    """Base class for all tools"""
    
    def info(self) -> ToolInfo:
        """Return tool information"""
        raise NotImplementedError
    
    async def run(self, params: ToolCall) -> ToolResponse:
        """Execute the tool"""
        raise NotImplementedError

class ViewTool(BaseTool):
    """Tool for viewing file contents"""
    
    def info(self) -> ToolInfo:
        return ToolInfo(
            name="view",
            description="View file contents with optional line range",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to view"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Starting line number (1-based)",
                        "default": 1
                    },
                    "limit": {
                        "type": "integer", 
                        "description": "Maximum number of lines to return",
                        "default": 1000
                    }
                },
                "required": ["file_path"]
            },
            required=["file_path"]
        )
    
    async def run(self, params: ToolCall) -> ToolResponse:
        try:
            args = json.loads(params.input)
            file_path = args["file_path"]
            offset = args.get("offset", 1)
            limit = args.get("limit", 1000)
            
            if not os.path.exists(file_path):
                return ToolResponse(
                    content=f"File not found: {file_path}",
                    is_error=True
                )
            
            if os.path.isdir(file_path):
                # List directory contents
                try:
                    items = []
                    for item in sorted(os.listdir(file_path)):
                        item_path = os.path.join(file_path, item)
                        if os.path.isdir(item_path):
                            items.append(f"{item}/")
                        else:
                            items.append(item)
                    return ToolResponse(
                        content=f"Directory listing for {file_path}:\n" + "\n".join(items)
                    )
                except PermissionError:
                    return ToolResponse(
                        content=f"Permission denied: {file_path}",
                        is_error=True
                    )
            
            # Read file contents
            try:
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    lines = await f.readlines()
                
                # Apply offset and limit
                start_idx = max(0, offset - 1)
                end_idx = min(len(lines), start_idx + limit)
                selected_lines = lines[start_idx:end_idx]
                
                # Add line numbers
                numbered_lines = []
                for i, line in enumerate(selected_lines, start=start_idx + 1):
                    numbered_lines.append(f"{i:4d} | {line.rstrip()}")
                
                return ToolResponse(
                    content="\n".join(numbered_lines),
                    metadata=json.dumps({
                        "file_path": file_path,
                        "total_lines": len(lines),
                        "displayed_lines": len(selected_lines),
                        "range": f"{start_idx + 1}-{end_idx}"
                    })
                )
                
            except UnicodeDecodeError:
                return ToolResponse(
                    content=f"Cannot read file (binary or unsupported encoding): {file_path}",
                    is_error=True
                )
            except PermissionError:
                return ToolResponse(
                    content=f"Permission denied: {file_path}",
                    is_error=True
                )
                
        except Exception as e:
            return ToolResponse(
                content=f"Error viewing file: {str(e)}",
                is_error=True
            )

class WriteTool(BaseTool):
    """Tool for writing content to files"""
    
    def info(self) -> ToolInfo:
        return ToolInfo(
            name="write",
            description="Write content to a file",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    },
                    "mode": {
                        "type": "string",
                        "description": "Write mode: 'w' (overwrite) or 'a' (append)",
                        "enum": ["w", "a"],
                        "default": "w"
                    }
                },
                "required": ["file_path", "content"]
            },
            required=["file_path", "content"]
        )
    
    async def run(self, params: ToolCall) -> ToolResponse:
        try:
            args = json.loads(params.input)
            file_path = args["file_path"]
            content = args["content"]
            mode = args.get("mode", "w")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            async with aiofiles.open(file_path, mode, encoding='utf-8') as f:
                await f.write(content)
            
            return ToolResponse(
                content=f"Successfully wrote to {file_path} ({len(content)} characters)",
                metadata=json.dumps({
                    "file_path": file_path,
                    "size": len(content),
                    "mode": mode
                })
            )
            
        except Exception as e:
            return ToolResponse(
                content=f"Error writing file: {str(e)}",
                is_error=True
            )

class EditTool(BaseTool):
    """Tool for editing files with search and replace"""
    
    def info(self) -> ToolInfo:
        return ToolInfo(
            name="edit",
            description="Edit file by replacing specific text",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to edit"
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Text to find and replace"
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Text to replace with"
                    },
                    "occurrence": {
                        "type": "integer",
                        "description": "Which occurrence to replace (0 for all, 1 for first, etc.)",
                        "default": 1
                    }
                },
                "required": ["file_path", "old_text", "new_text"]
            },
            required=["file_path", "old_text", "new_text"]
        )
    
    async def run(self, params: ToolCall) -> ToolResponse:
        try:
            args = json.loads(params.input)
            file_path = args["file_path"]
            old_text = args["old_text"]
            new_text = args["new_text"]
            occurrence = args.get("occurrence", 1)
            
            if not os.path.exists(file_path):
                return ToolResponse(
                    content=f"File not found: {file_path}",
                    is_error=True
                )
            
            # Read file content
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Perform replacement
            if occurrence == 0:
                # Replace all occurrences
                new_content = content.replace(old_text, new_text)
                replacements = content.count(old_text)
            else:
                # Replace specific occurrence
                parts = content.split(old_text)
                if len(parts) < occurrence + 1:
                    return ToolResponse(
                        content=f"Occurrence {occurrence} not found in {file_path}",
                        is_error=True
                    )
                
                new_content = old_text.join(parts[:occurrence]) + new_text + old_text.join(parts[occurrence:])
                replacements = 1
            
            if replacements == 0:
                return ToolResponse(
                    content=f"Text not found in {file_path}: {old_text}",
                    is_error=True
                )
            
            # Write back to file
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(new_content)
            
            return ToolResponse(
                content=f"Successfully replaced {replacements} occurrence(s) in {file_path}",
                metadata=json.dumps({
                    "file_path": file_path,
                    "replacements": replacements,
                    "old_text": old_text[:100] + "..." if len(old_text) > 100 else old_text,
                    "new_text": new_text[:100] + "..." if len(new_text) > 100 else new_text
                })
            )
            
        except Exception as e:
            return ToolResponse(
                content=f"Error editing file: {str(e)}",
                is_error=True
            )

class GlobTool(BaseTool):
    """Tool for finding files by pattern"""
    
    def info(self) -> ToolInfo:
        return ToolInfo(
            name="glob",
            description="Find files matching a glob pattern",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., '*.py', '**/*.js')"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in",
                        "default": "."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 100
                    }
                },
                "required": ["pattern"]
            },
            required=["pattern"]
        )
    
    async def run(self, params: ToolCall) -> ToolResponse:
        try:
            args = json.loads(params.input)
            pattern = args["pattern"]
            search_path = args.get("path", ".")
            max_results = args.get("max_results", 100)
            
            matches = []
            
            if "**" in pattern:
                # Recursive pattern
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, search_path)
                        if fnmatch.fnmatch(rel_path, pattern):
                            matches.append(rel_path)
                            if len(matches) >= max_results:
                                break
                    if len(matches) >= max_results:
                        break
            else:
                # Non-recursive pattern
                import glob
                matches = glob.glob(os.path.join(search_path, pattern))
                matches = [os.path.relpath(m, search_path) for m in matches]
            
            matches = sorted(matches[:max_results])
            
            if not matches:
                return ToolResponse(
                    content=f"No files found matching pattern: {pattern}"
                )
            
            return ToolResponse(
                content=f"Found {len(matches)} file(s) matching '{pattern}':\n" + "\n".join(matches),
                metadata=json.dumps({
                    "pattern": pattern,
                    "path": search_path,
                    "count": len(matches),
                    "files": matches
                })
            )
            
        except Exception as e:
            return ToolResponse(
                content=f"Error finding files: {str(e)}",
                is_error=True
            )

class GrepTool(BaseTool):
    """Tool for searching file contents"""
    
    def info(self) -> ToolInfo:
        return ToolInfo(
            name="grep",
            description="Search for text patterns in files",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in",
                        "default": "."
                    },
                    "include": {
                        "type": "string",
                        "description": "File pattern to include (e.g., '*.py')"
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Number of context lines around matches",
                        "default": 0
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 50
                    }
                },
                "required": ["pattern"]
            },
            required=["pattern"]
        )
    
    async def run(self, params: ToolCall) -> ToolResponse:
        try:
            args = json.loads(params.input)
            pattern = args["pattern"]
            search_path = args.get("path", ".")
            include_pattern = args.get("include", "*")
            context_lines = args.get("context_lines", 0)
            max_results = args.get("max_results", 50)
            
            results = []
            result_count = 0
            
            # Compile regex pattern
            try:
                regex = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                return ToolResponse(
                    content=f"Invalid regex pattern: {str(e)}",
                    is_error=True
                )
            
            if os.path.isfile(search_path):
                # Search in single file
                await self._search_file(search_path, regex, context_lines, results)
            else:
                # Search in directory
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        if fnmatch.fnmatch(file, include_pattern):
                            file_path = os.path.join(root, file)
                            try:
                                matches = await self._search_file(file_path, regex, context_lines, [])
                                results.extend(matches)
                                result_count += len(matches)
                                if result_count >= max_results:
                                    break
                            except:
                                continue
                    if result_count >= max_results:
                        break
            
            results = results[:max_results]
            
            if not results:
                return ToolResponse(
                    content=f"No matches found for pattern: {pattern}"
                )
            
            return ToolResponse(
                content=f"Found {len(results)} match(es) for '{pattern}':\n\n" + "\n\n".join(results),
                metadata=json.dumps({
                    "pattern": pattern,
                    "path": search_path,
                    "count": len(results)
                })
            )
            
        except Exception as e:
            return ToolResponse(
                content=f"Error searching files: {str(e)}",
                is_error=True
            )
    
    async def _search_file(self, file_path: str, regex: re.Pattern, context_lines: int, results: List[str]) -> List[str]:
        """Search for pattern in a single file"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            lines = content.splitlines()
            matches = []
            
            for i, line in enumerate(lines):
                if regex.search(line):
                    # Build context
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    
                    context = []
                    for j in range(start, end):
                        prefix = ">" if j == i else " "
                        context.append(f"{prefix} {j+1:4d}: {lines[j]}")
                    
                    match_text = f"{file_path}:\n" + "\n".join(context)
                    matches.append(match_text)
            
            return matches
            
        except (UnicodeDecodeError, PermissionError):
            return []

class BashTool(BaseTool):
    """Tool for executing bash commands"""
    
    def info(self) -> ToolInfo:
        return ToolInfo(
            name="bash",
            description="Execute bash commands",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Command timeout in seconds",
                        "default": 30
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory for the command",
                        "default": "."
                    }
                },
                "required": ["command"]
            },
            required=["command"]
        )
    
    async def run(self, params: ToolCall) -> ToolResponse:
        try:
            args = json.loads(params.input)
            command = args["command"]
            timeout = args.get("timeout", 30)
            working_dir = args.get("working_dir", ".")
            
            # Security check - basic command validation
            dangerous_commands = ['rm -rf', 'format', 'del /f', 'sudo rm', 'chmod 777']
            if any(dangerous in command.lower() for dangerous in dangerous_commands):
                return ToolResponse(
                    content=f"Command rejected for security reasons: {command}",
                    is_error=True
                )
            
            # Execute command
            try:
                result = await asyncio.wait_for(
                    asyncio.create_subprocess_shell(
                        command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=working_dir
                    ),
                    timeout=timeout
                )
                
                stdout, stderr = await result.communicate()
                
                output = stdout.decode('utf-8', errors='replace')
                error = stderr.decode('utf-8', errors='replace')
                
                if result.returncode == 0:
                    return ToolResponse(
                        content=output or "Command completed successfully (no output)",
                        metadata=json.dumps({
                            "command": command,
                            "return_code": result.returncode,
                            "working_dir": working_dir
                        })
                    )
                else:
                    return ToolResponse(
                        content=f"Command failed (exit code {result.returncode}):\n{error}",
                        is_error=True,
                        metadata=json.dumps({
                            "command": command,
                            "return_code": result.returncode,
                            "stderr": error
                        })
                    )
                    
            except asyncio.TimeoutError:
                return ToolResponse(
                    content=f"Command timed out after {timeout} seconds",
                    is_error=True
                )
                
        except Exception as e:
            return ToolResponse(
                content=f"Error executing command: {str(e)}",
                is_error=True
            )

class FetchTool(BaseTool):
    """Tool for fetching data from URLs"""
    
    def info(self) -> ToolInfo:
        return ToolInfo(
            name="fetch",
            description="Fetch data from URLs",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch"
                    },
                    "method": {
                        "type": "string",
                        "description": "HTTP method",
                        "enum": ["GET", "POST", "PUT", "DELETE"],
                        "default": "GET"
                    },
                    "headers": {
                        "type": "object",
                        "description": "HTTP headers"
                    },
                    "data": {
                        "type": "string",
                        "description": "Request body data"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Request timeout in seconds",
                        "default": 30
                    }
                },
                "required": ["url"]
            },
            required=["url"]
        )
    
    async def run(self, params: ToolCall) -> ToolResponse:
        try:
            args = json.loads(params.input)
            url = args["url"]
            method = args.get("method", "GET")
            headers = args.get("headers", {})
            data = args.get("data")
            timeout = args.get("timeout", 30)
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    content=data
                )
                
                content_type = response.headers.get('content-type', '')
                
                if 'application/json' in content_type:
                    content = json.dumps(response.json(), indent=2)
                else:
                    content = response.text
                
                return ToolResponse(
                    content=f"HTTP {response.status_code} from {url}:\n\n{content}",
                    metadata=json.dumps({
                        "url": url,
                        "status_code": response.status_code,
                        "headers": dict(response.headers),
                        "content_type": content_type
                    })
                )
                
        except Exception as e:
            return ToolResponse(
                content=f"Error fetching URL: {str(e)}",
                is_error=True
            )

# Tool registry
AVAILABLE_TOOLS = {
    "view": ViewTool(),
    "write": WriteTool(),
    "edit": EditTool(),
    "glob": GlobTool(),
    "grep": GrepTool(),
    "bash": BashTool(),
    "fetch": FetchTool()
}

def get_tool(name: str) -> Optional[BaseTool]:
    """Get a tool by name"""
    return AVAILABLE_TOOLS.get(name)

def get_all_tools() -> List[BaseTool]:
    """Get all available tools"""
    return list(AVAILABLE_TOOLS.values())

def get_tool_schemas() -> List[Dict[str, Any]]:
    """Get OpenAI-compatible tool schemas"""
    schemas = []
    for tool in AVAILABLE_TOOLS.values():
        info = tool.info()
        schemas.append({
            "type": "function",
            "function": {
                "name": info.name,
                "description": info.description,
                "parameters": info.parameters
            }
        })
    return schemas