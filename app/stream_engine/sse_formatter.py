"""SSE格式化器，创建OpenAI兼容的事件格式"""

import json
import time
from typing import Dict, Any, Optional


class SSEFormatter:
    """SSE格式化器"""
    
    FINISH_REASON_MAP = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "stop",
        "OTHER": "stop",
    }
    
    def __init__(self, conversation_id: str):
        self._conversation_id = conversation_id
    
    def _generate_conversation_chunk_id(self) -> str:
        return f"chatcmpl-{self._conversation_id[:8]}"
    
    def format_sse_event(
        self,
        data: Dict[str, Any],
        event_id: Optional[str] = None,
        event_type: Optional[str] = None
    ) -> str:
        """格式化为SSE事件"""
        data_json = json.dumps(data, ensure_ascii=False)
        return f"data: {data_json}\n\n"
    
    def create_heartbeat_event(self, sequence: int) -> str:
        """创建心跳事件（空delta的OpenAI chunk）"""
        chunk = {
            "id": self._generate_conversation_chunk_id(),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "vertex-ai-proxy",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": None
            }]
        }
        return self.format_sse_event(data=chunk)
    
    def create_openai_chunk(
        self,
        content: Optional[str] = None,
        reasoning_content: Optional[str] = None,
        finish_reason: Optional[str] = None,
        model: str = "vertex-ai-proxy",
        include_role: bool = False
    ) -> Dict[str, Any]:
        """创建OpenAI格式的chunk"""
        delta = {}
        
        if include_role:
            delta["role"] = "assistant"
        
        if reasoning_content is not None:
            delta["reasoning_content"] = reasoning_content
        
        if content is not None:
            delta["content"] = content
        
        chunk = {
            "id": self._generate_conversation_chunk_id(),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }]
        }
        
        return chunk
    
    def create_initial_role_chunk(self, model: str = "vertex-ai-proxy") -> str:
        """创建包含role的初始chunk"""
        chunk = {
            "id": self._generate_conversation_chunk_id(),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None
            }]
        }
        return self.format_sse_event(chunk)
    
    def map_finish_reason(self, vertex_reason: str) -> str:
        """Vertex AI finish reason -> OpenAI格式"""
        return self.FINISH_REASON_MAP.get(vertex_reason, "stop")