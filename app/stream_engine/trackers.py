"""流式数据追踪器：状态枚举、索引追踪、缓冲区"""

import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from threading import Lock
from enum import Enum


class DiffState(Enum):
    """Diff块处理状态"""
    NORMAL = "normal"           # 普通文本模式
    IN_DIFF = "in_diff"         # 在diff块内部


@dataclass
class PathIndexTracker:
    """
    基于Path索引的增量追踪器
    
    使用 (path_index, is_thought) 复合key分别追踪思考内容和正常内容，
    通过长度比较自动去重，回退时保持已提交进度不变。
    """
    path_content: Dict[Tuple[int, bool], str] = field(default_factory=dict)
    committed_length: Dict[Tuple[int, bool], int] = field(default_factory=dict)
    emitted_length: Dict[Tuple[int, bool], int] = field(default_factory=dict)
    last_processed_index: int = -1
    global_sequence: int = 0
    backtrack_events: int = 0
    duplicate_events: int = 0
    content_updates: int = 0
    out_of_order_events: int = 0
    # 锁
    _lock: Lock = field(default_factory=Lock)
    
    def process_result(self, path_index: int, text: str, is_thought: bool = False) -> Optional[Tuple[int, str, bool]]:
        """处理result，返回增量内容或None"""
        with self._lock:
            if path_index < 0:
                return None
            
            composite_key = (path_index, is_thought)
            
            if composite_key not in self.path_content:
                self.path_content[composite_key] = ""
                self.committed_length[composite_key] = 0
                self.emitted_length[composite_key] = 0
            
            if not is_thought and path_index < self.last_processed_index:
                self.out_of_order_events += 1
            
            current_len = len(text)
            committed = self.committed_length[composite_key]
            
            if current_len > committed:
                delta = text[committed:]
                self.path_content[composite_key] = text
                self.committed_length[composite_key] = current_len
                self.emitted_length[composite_key] = current_len
                if not is_thought:
                    self.last_processed_index = max(self.last_processed_index, path_index)
                self.global_sequence += 1
                self.content_updates += 1
                return (path_index, delta, is_thought)
                
            elif current_len < committed:
                # 回退检测：保持已提交进度不变
                self.backtrack_events += 1
                self.path_content[composite_key] = text
                thought_tag = "[思考]" if is_thought else "[正文]"
                print(f"⚠️ [PathIndexTracker] {thought_tag} path={path_index} 回退: committed={committed}, 新长度={current_len}, 保持committed不变")
                return None
                
            else:
                self.duplicate_events += 1
                return None
    
    def get_pending_content(self) -> List[Tuple[int, str, bool]]:
        """获取所有未发送的待处理内容"""
        with self._lock:
            pending = []
            for composite_key in sorted(self.path_content.keys(), key=lambda k: (not k[1], k[0])):
                path_index, is_thought = composite_key
                content = self.path_content[composite_key]
                emitted = self.emitted_length.get(composite_key, 0)
                if len(content) > emitted:
                    delta = content[emitted:]
                    pending.append((path_index, delta, is_thought))
                    self.emitted_length[composite_key] = len(content)
            return pending
    
    def get_stats(self) -> Dict[str, Any]:
        """获取追踪器统计信息"""
        with self._lock:
            total_content = sum(len(c) for c in self.path_content.values())
            total_committed = sum(self.committed_length.values())
            total_emitted = sum(self.emitted_length.values())
            thought_content_len = sum(len(c) for k, c in self.path_content.items() if k[1])
            normal_content_len = sum(len(c) for k, c in self.path_content.items() if not k[1])
            return {
                "tracked_paths": len(self.path_content),
                "total_content_length": total_content,
                "thought_content_length": thought_content_len,
                "normal_content_length": normal_content_len,
                "total_committed_length": total_committed,
                "total_emitted_length": total_emitted,
                "last_processed_index": self.last_processed_index,
                "global_sequence": self.global_sequence,
                "content_updates": self.content_updates,
                "backtrack_events": self.backtrack_events,
                "duplicate_events": self.duplicate_events,
                "out_of_order_events": self.out_of_order_events
            }


@dataclass
class StreamBuffer:
    """简化的流式缓冲区"""
    sequence_counter: int = 0
    last_yield_time: float = field(default_factory=time.time)
    total_content_length: int = 0
    chunks_yielded: int = 0
    _lock: Lock = field(default_factory=Lock)
    
    def mark_content_sent(self, content: str):
        with self._lock:
            self.total_content_length += len(content)
            self.chunks_yielded += 1
    
    def should_send_heartbeat(self, interval: float = 15.0) -> bool:
        with self._lock:
            return (time.time() - self.last_yield_time) >= interval
    
    def mark_yield(self):
        with self._lock:
            self.last_yield_time = time.time()
    
    def increment_sequence(self) -> int:
        with self._lock:
            self.sequence_counter += 1
            return self.sequence_counter
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "sequence": self.sequence_counter,
                "total_content_length": self.total_content_length,
                "chunks_yielded": self.chunks_yielded
            }