"""Diff块处理器，确保diff块原子传输"""

from typing import List, Tuple, Optional

from .trackers import DiffState


class DiffBlockHandler:
    """
    Diff块处理器
    
    专门处理Vibe Coding工具的diff格式：
    <<<<<<< SEARCH
    原始内容
    =======
    替换内容
    >>>>>>> REPLACE
    
    确保diff块作为原子单元传输，不会被拆分
    """
    
    # Diff标记
    SEARCH_START = "<<<<<<< SEARCH"
    REPLACE_END = ">>>>>>> REPLACE"
    
    def __init__(self):
        self.state = DiffState.NORMAL
        self.diff_buffer = ""
        self.pending_buffer = ""
    
    def _find_partial_match(self, text: str, marker: str) -> int:
        """查找部分标记匹配长度"""
        for i in range(len(marker) - 1, 0, -1):
            if text.endswith(marker[:i]):
                return i
        return 0
    
    def process(self, text: str) -> List[Tuple[str, bool]]:
        """处理文本，返回 (内容, 是否为diff块) 列表"""
        results = []
        self.pending_buffer += text
        
        while self.pending_buffer:
            if self.state == DiffState.NORMAL:
                search_pos = self.pending_buffer.find(self.SEARCH_START)
                
                if search_pos != -1:
                    if search_pos > 0:
                        results.append((self.pending_buffer[:search_pos], False))
                    
                    self.pending_buffer = self.pending_buffer[search_pos:]
                    self.state = DiffState.IN_DIFF
                    continue
                
                else:
                    partial_len = self._find_partial_match(self.pending_buffer, self.SEARCH_START)
                    
                    if partial_len > 0:
                        keep_len = partial_len
                    else:
                        keep_len = min(len(self.SEARCH_START) - 1, len(self.pending_buffer))
                    
                    if len(self.pending_buffer) > keep_len:
                        safe_len = len(self.pending_buffer) - keep_len
                        safe_content = self.pending_buffer[:safe_len]
                        results.append((safe_content, False))
                        self.pending_buffer = self.pending_buffer[safe_len:]
                    
                    break
            
            elif self.state == DiffState.IN_DIFF:
                replace_pos = self.pending_buffer.find(self.REPLACE_END)
                
                if replace_pos != -1:
                    end_pos = replace_pos + len(self.REPLACE_END)
                    self.diff_buffer += self.pending_buffer[:end_pos]
                    results.append((self.diff_buffer, True))
                    
                    self.state = DiffState.NORMAL
                    self.diff_buffer = ""
                    self.pending_buffer = self.pending_buffer[end_pos:]
                    continue
                
                else:
                    partial_len = self._find_partial_match(self.pending_buffer, self.REPLACE_END)
                    
                    if partial_len > 0:
                        keep_len = partial_len
                    else:
                        keep_len = min(len(self.REPLACE_END) - 1, len(self.pending_buffer))
                    
                    if len(self.pending_buffer) > keep_len:
                        safe_len = len(self.pending_buffer) - keep_len
                        self.diff_buffer += self.pending_buffer[:safe_len]
                        self.pending_buffer = self.pending_buffer[safe_len:]
                    
                    break
        
        return results
    
    def flush(self) -> Optional[Tuple[str, bool]]:
        """刷新缓冲区"""
        if self.state == DiffState.NORMAL:
            if self.pending_buffer:
                content = self.pending_buffer
                self.pending_buffer = ""
                return (content, False)
        elif self.state == DiffState.IN_DIFF:
            content = self.diff_buffer + self.pending_buffer
            self.diff_buffer = ""
            self.pending_buffer = ""
            self.state = DiffState.NORMAL
            return (content, False)
            
        return None

    def flush_pending(self) -> Optional[Tuple[str, bool]]:
        """强制刷新pending_buffer"""
        if self.state == DiffState.NORMAL and self.pending_buffer:
            if self.SEARCH_START.startswith(self.pending_buffer) or self.pending_buffer.startswith("<"):
                if len(self.pending_buffer) < len(self.SEARCH_START):
                    return None
            
            content = self.pending_buffer
            self.pending_buffer = ""
            return (content, False)
        return None
    
    def is_in_diff(self) -> bool:
        return self.state == DiffState.IN_DIFF