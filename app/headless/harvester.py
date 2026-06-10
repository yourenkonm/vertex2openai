"""
凭证抓取模块

从无头浏览器拦截的请求中提取凭证信息。
"""

import json
import time
from typing import Dict, Any, Optional, Callable


class CredentialHarvester:
    """凭证抓取器 - 从浏览器请求中提取 auth headers/cookies/url/body"""

    # 目标请求 URL 特征
    TARGET_PATTERNS = [
        "batchGraphql",
        "StreamGenerateContent",
    ]

    # 需要提取的重要 Headers
    IMPORTANT_HEADERS = [
        "authorization",
        "x-goog-authuser",
        "x-goog-first-party-reauth",
        "x-origin",
        "origin",
        "referer",
        "x-same-domain",
        "cookie",
        "user-agent",
    ]

    def __init__(self, on_credentials: Optional[Callable] = None):
        """
        Args:
            on_credentials: 获取到凭证时的回调函数，接收凭证 dict
        """
        self.on_credentials = on_credentials
        self._last_credentials: Optional[Dict[str, Any]] = None
        self._capture_count = 0

    def is_target_request(self, url: str) -> bool:
        """检查是否为目标请求"""
        return any(pattern in url for pattern in self.TARGET_PATTERNS)

    async def handle_request(self, request) -> None:
        """
        处理拦截到的请求

        从 Playwright Request 对象中提取凭证，过滤掉非内容生成请求。

        Args:
            request: Playwright Request 对象
        """
        url = request.url

        if not self.is_target_request(url):
            return

        try:
            # 提取所有 Headers
            all_headers = await request.all_headers()
            headers = {}

            # 过滤出重要的 Headers（大小写不敏感匹配）
            for key in self.IMPORTANT_HEADERS:
                for h_key, h_value in all_headers.items():
                    if h_key.lower() == key.lower():
                        headers[h_key] = h_value
                        break

            # 提取 Cookie
            cookies = all_headers.get("cookie", "")

            # 提取请求体
            body = None
            post_data_str = ""
            try:
                post_data = request.post_data
                if post_data:
                    post_data_str = post_data
                    body = json.loads(post_data)
            except (json.JSONDecodeError, TypeError):
                pass

            # 关键过滤：只捕获实际的内容生成请求，跳过 UI 状态请求
            content_keywords = ["StreamGenerateContent", "generateContent", "Predict", "Image"]
            is_content_request = any(kw in post_data_str for kw in content_keywords)

            if not is_content_request:
                # 这是 UI 状态请求（如 batchGraphql 拉取界面数据），不捕获
                return

            # 创建凭证字典 - 兼容 app_state.update_auth_bundle() 格式
            credentials = {
                "headers": headers,
                "cookies": cookies,
                "url": url,
                "body": body,
                "timestamp": time.time(),
            }

            self._last_credentials = credentials
            self._capture_count += 1

            print(f"🎯 捕获凭证 #{self._capture_count}")
            print(f"   URL: {url[:80]}...")
            print(f"   Headers: {len(headers)} 个")

            # 调用回调通知上层
            if self.on_credentials:
                await self._call_callback(credentials)

        except Exception as e:
            print(f"⚠️ 处理请求时出错: {e}")

    async def _call_callback(self, credentials: Dict[str, Any]) -> None:
        """调用凭证回调，支持同步和异步回调"""
        try:
            result = self.on_credentials(credentials)
            # 支持异步回调
            if hasattr(result, "__await__"):
                await result
        except Exception as e:
            print(f"⚠️ 凭证回调出错: {e}")

    def get_credentials(self) -> Optional[Dict[str, Any]]:
        """获取最新凭证"""
        return self._last_credentials

    @property
    def capture_count(self) -> int:
        return self._capture_count
