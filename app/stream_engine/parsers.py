"""增量JSON解析器，处理流式传输中的不完整数据"""

import json
from typing import Dict, Any, List


class IncrementalJSONParser:
    """增量JSON解析器，支持NDJSON和JSON数组格式"""
    
    def __init__(self):
        self.buffer = ""
        self.decoder = json.JSONDecoder()
        self.parse_errors = 0
        self.successful_parses = 0
    
    def _is_likely_complete_json(self, text: str) -> bool:
        """快速检查是否可能是完整JSON"""
        if not text:
            return False
        
        text = text.strip()
        if not text:
            return False
        
        if text[0] not in '{[':
            return False
        
        brace_count = text.count('{') - text.count('}')
        bracket_count = text.count('[') - text.count(']')
        
        if brace_count != 0 or bracket_count != 0:
            return False
        
        return True
    
    def feed(self, data: str) -> List[Dict[str, Any]]:
        """输入数据并返回所有可解析的JSON对象"""
        self.buffer += data
        results = []
        
        lines = self.buffer.split('\n')
        
        if len(lines) > 1:
            self.buffer = lines[-1]
            lines = lines[:-1]
        else:
            lines = []
        
        for line in lines:
            line = line.strip()
            if not line or line in [',', '[', ']']:
                continue
            
            line = line.strip(',')
            if not line:
                continue
                
            try:
                obj = json.loads(line)
                results.append(obj)
                self.successful_parses += 1
            except json.JSONDecodeError:
                self.buffer = line + '\n' + self.buffer
                self.parse_errors += 1
                break
        
        if not results and self.buffer:
            while self.buffer:
                self.buffer = self.buffer.lstrip()
                if not self.buffer:
                    break
                
                if self.buffer[0] in '[,]':
                    self.buffer = self.buffer[1:]
                    continue
                
                if not self._is_likely_complete_json(self.buffer.split('\n')[0] if '\n' in self.buffer else self.buffer):
                    break
                
                try:
                    obj, idx = self.decoder.raw_decode(self.buffer)
                    results.append(obj)
                    self.successful_parses += 1
                    self.buffer = self.buffer[idx:]
                except json.JSONDecodeError:
                    self.parse_errors += 1
                    break
        
        return results
    
    def get_remaining(self) -> str:
        return self.buffer
        
    def flush(self) -> List[Dict[str, Any]]:
        """强制刷新缓冲区"""
        results = []
        if self.buffer:
            try:
                self.buffer = self.buffer.lstrip()
                if self.buffer:
                    obj, idx = self.decoder.raw_decode(self.buffer)
                    results.append(obj)
                    self.successful_parses += 1
                    self.buffer = self.buffer[idx:]
            except json.JSONDecodeError:
                self.parse_errors += 1
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "buffer_length": len(self.buffer),
            "successful_parses": self.successful_parses,
            "parse_errors": self.parse_errors
        }
    
    def clear(self):
        self.buffer = ""