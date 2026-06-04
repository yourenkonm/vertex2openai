import json
import time
import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

from models import OpenAIRequest
from upstreams.base import BaseUpstream
from upstreams.studio_payload import build_studio_graphql_payload
from runtime_state import app_state

# 引入 11-30 完美的流式追踪与消抖处理器
from stream_engine.processor import StreamProcessor


class WebProxyUpstream(BaseUpstream):
    """
    谷歌 Agent Platform Studio 网页反代渠道处理器
    封装了动态 Payload 构造、HTTP/2 双向隧道以及非流式聚合拼装逻辑
    """
    async def chat_completions(self, request_obj: OpenAIRequest, fastapi_request: Request):
        auth_bundle = app_state.get_auth_bundle()
        if not auth_bundle or "headers" not in auth_bundle:
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Web Proxy 凭证尚未配置，请在控制台填入最新 Auth Bundle", "type": "auth_error"}}
            )

        # 1. 规范化模型名称，如果是 -search 则在 payload 中启用 search_tool
        base_model_name = request_obj.model
        is_search = False
        if base_model_name.endswith("-search"):
            base_model_name = base_model_name[:-len("-search")]
            is_search = True

        from api_helpers import create_generation_config
        gen_config_dict = create_generation_config(request_obj)

        # 2. 动态拼装 GraphQL 载荷
        payload = build_studio_graphql_payload(base_model_name, request_obj, gen_config_dict, auth_bundle)
        
        # 激活谷歌搜索 Grounding 插件
        if is_search:
            payload["variables"].setdefault("tools", []).append({"googleSearch": {}})
            print(f"🔎 [搜索增强] 已为 Web 模式下的模型 {base_model_name} 挂载 googleSearch 插件。")

        url = auth_bundle.get("url")
        headers = auth_bundle.get("headers", {}).copy()
        
        # 清除会导致 httpx 双流异步解析异常的 Header
        headers.pop("accept-encoding", None)
        headers.pop("content-length", None)
        headers["content-type"] = "application/json"

        # 3. 流式处理通道 (stream = True)
        if request_obj.stream:
            async def stream_generator():
                processor = StreamProcessor()
                # 必须指定 http2=True，谷歌后台 GraphQL 的高速并发强依赖 HTTP/2 多路复用机制
                async with httpx.AsyncClient(http2=True, timeout=120.0) as client:
                    try:
                        async with client.stream("POST", url, headers=headers, json=payload) as response:
                            if response.status_code != 200:
                                error_text = await response.aread()
                                yield f"data: {json.dumps({'error': f'Studio Error {response.status_code}: {error_text.decode()}'})}\n\n"
                                return
                            
                            async for sse_event in processor.process_stream(response.aiter_text(), model=request_obj.model):
                                yield sse_event
                    except Exception as e:
                        yield f"data: {json.dumps({'error': f'Stream translation failed: {str(e)}'})}\n\n"
            
            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        # 4. 非流式处理通道 (stream = False)，在后端自动聚合 GraphQL 流
        else:
            full_text = ""
            reasoning_text = ""
            final_finish_reason = "stop"
            tool_calls = []
            
            processor = StreamProcessor()
            async with httpx.AsyncClient(http2=True, timeout=120.0) as client:
                try:
                    async with client.stream("POST", url, headers=headers, json=payload) as response:
                        if response.status_code != 200:
                            error_text = await response.aread()
                            return JSONResponse(status_code=response.status_code, content={"error": error_text.decode()})
                        
                        # 循环读取流，并实时解包拼接
                        async for sse_event in processor.process_stream(response.aiter_text(), model=request_obj.model):
                            if sse_event.startswith("data: "):
                                data_str = sse_event[6:].strip()
                                if data_str == "[DONE]":
                                    continue
                                try:
                                    chunk = json.loads(data_str)
                                    choices = chunk.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        if "content" in delta and delta["content"] is not None:
                                            full_text += delta["content"]
                                        if "reasoning_content" in delta and delta["reasoning_content"] is not None:
                                            reasoning_text += delta["reasoning_content"]
                                        if "tool_calls" in delta and delta["tool_calls"] is not None:
                                            # 处理并缝合流式传输工具调用片段 (Stitch)
                                            for tc_delta in delta["tool_calls"]:
                                                idx = tc_delta.get("index", 0)
                                                if len(tool_calls) <= idx:
                                                    tool_calls.append({
                                                        "id": tc_delta.get("id", ""),
                                                        "type": "function",
                                                        "function": {"name": "", "arguments": ""}
                                                    })
                                                tc = tool_calls[idx]
                                                if tc_delta.get("id"):
                                                    tc["id"] = tc_delta["id"]
                                                if "function" in tc_delta:
                                                    fn_delta = tc_delta["function"]
                                                    if "name" in fn_delta:
                                                        tc["function"]["name"] = fn_delta["name"]
                                                    if "arguments" in fn_delta:
                                                        tc["function"]["arguments"] += fn_delta["arguments"]
                                        if choices[0].get("finish_reason"):
                                            final_finish_reason = choices[0]["finish_reason"]
                                except Exception:
                                    pass
                except Exception as e:
                    return JSONResponse(status_code=500, content={"error": f"Failed to gather studio response: {str(e)}"})

            # 重组标准 OpenAI ChatCompletion 数据包
            message_payload = {"role": "assistant"}
            if tool_calls:
                message_payload["tool_calls"] = tool_calls
                message_payload["content"] = None
            else:
                message_payload["content"] = full_text
                if reasoning_text:
                    message_payload["reasoning_content"] = reasoning_text
                    
            return JSONResponse(content={
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request_obj.model,
                "choices": [{
                    "index": 0,
                    "message": message_payload,
                    "finish_reason": final_finish_reason
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            })