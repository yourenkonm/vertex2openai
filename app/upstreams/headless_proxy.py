"""
无头浏览器 / Cookie 直连 代理上游通道

支持两种模式：
1. Cookie 直连模式（推荐）：从环境变量或大盘粘贴的 Cookie 中计算 SAPISIDHASH，直接调用 API
2. 凭证捕获模式：从无头浏览器截获的凭证中复用请求

两种模式对调用方完全透明，自动选择可用的认证方式。
"""

import copy
import json
import time
import httpx
import traceback
from typing import Any
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from models import OpenAIRequest
from upstreams.base import BaseUpstream
from runtime_state import app_state
import config as app_config

# Cookie 直连认证
from cookie_auth import build_auth_headers, build_vertex_url, extract_sapisid

# 引入 google-genai 类型库和 OpenAI 格式转换器
from google.genai import types
from api_helpers import convert_chunk_to_openai
from message_processing import create_gemini_prompt

# 引入流式追踪与消抖处理器 (用于兼容 GraphQL 旧接口)
from stream_engine.processor import StreamProcessor


# ========== 载荷构建工具函数 ==========

def _serialize_pydantic(obj: Any) -> Any:
    """将 Pydantic 模型递归序列化为 dict"""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    elif isinstance(obj, dict):
        return {k: _serialize_pydantic(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_pydantic(x) for x in obj]
    return obj


_KEY_MAP = {
    "max_output_tokens": "maxOutputTokens",
    "stop_sequences": "stopSequences",
    "top_p": "topP",
    "top_k": "topK",
    "candidate_count": "candidateCount",
    "presence_penalty": "presencePenalty",
    "frequency_penalty": "frequencyPenalty",
    "response_mime_type": "responseMimeType",
    "thinking_config": "thinkingConfig",
    "include_thoughts": "includeThoughts",
    "thinking_budget": "thinkingBudget",
    "thinking_level": "thinkingLevel",
    "image_config": "imageConfig",
    "image_size": "imageSize",
    "aspect_ratio": "aspectRatio",
    "safety_settings": "safetySettings",
    "system_instruction": "systemInstruction",
    "inline_data": "inlineData",
    "mime_type": "mimeType",
    "function_call": "functionCall",
    "function_response": "functionResponse"
}


def _convert_keys_to_camel(obj: Any) -> Any:
    """snake_case 键名转 camelCase"""
    if isinstance(obj, dict):
        return {_KEY_MAP.get(k, k): _convert_keys_to_camel(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_keys_to_camel(x) for x in obj]
    return obj


def _build_fresh_payload(model_name: str, request: OpenAIRequest) -> dict:
    """
    从零构建 API 请求载荷（Cookie 直连模式专用）
    不依赖任何捕获的请求体模板
    """
    raw_contents = create_gemini_prompt(request.messages)
    camel_contents = _convert_keys_to_camel(_serialize_pydantic(raw_contents))

    system_texts = [m.content for m in request.messages if m.role == "system" and isinstance(m.content, str)]

    payload = {"contents": camel_contents}

    # 生成配置
    gc = {}
    if request.temperature is not None: gc["temperature"] = request.temperature
    if request.max_tokens is not None: gc["maxOutputTokens"] = request.max_tokens
    if request.top_p is not None: gc["topP"] = request.top_p
    if request.stop is not None: gc["stopSequences"] = request.stop

    # 思考模式
    if "gemini-3" in model_name or "gemini-2.5" in model_name:
        gc["thinkingConfig"] = {"includeThoughts": True, "thinkingLevel": "MEDIUM"}

    if gc:
        payload["generationConfig"] = gc

    # 系统提示
    if system_texts:
        payload["systemInstruction"] = {"parts": [{"text": "\n".join(system_texts)}]}

    return payload


def _build_payload_from_template(model_name: str, request: OpenAIRequest, template_body: dict) -> dict:
    """
    基于捕获的请求体模板构建载荷（浏览器捕获模式）
    """
    payload = copy.deepcopy(template_body)

    raw_contents = create_gemini_prompt(request.messages)
    camel_contents = _convert_keys_to_camel(_serialize_pydantic(raw_contents))
    system_texts = [m.content for m in request.messages if m.role == "system" and isinstance(m.content, str)]

    # 模式 A：旧版 batchGraphql 格式 (含有 variables 节点)
    if "variables" in payload:
        variables = payload["variables"]
        variables["contents"] = camel_contents

        harvested_model = variables.get("model", "")
        if harvested_model and "/" in harvested_model:
            parts = harvested_model.split("/")
            parts[-1] = model_name
            variables["model"] = "/".join(parts)
        else:
            variables["model"] = model_name

        if "generationConfig" not in variables:
            variables["generationConfig"] = {}
        gc = variables["generationConfig"]
        if request.temperature is not None: gc["temperature"] = request.temperature
        if request.max_tokens is not None: gc["maxOutputTokens"] = request.max_tokens
        if request.top_p is not None: gc["topP"] = request.top_p
        if request.stop is not None: gc["stopSequences"] = request.stop
        if "gemini-3" in model_name or "gemini-2.5" in model_name:
            gc["thinkingConfig"] = {"includeThoughts": True, "thinkingLevel": "MEDIUM"}
        else:
            gc.pop("thinkingConfig", None)
        if system_texts:
            variables["systemInstruction"] = {"parts": [{"text": "\n".join(system_texts)}]}

    # 模式 B：标准 REST streamGenerateContent 格式
    else:
        payload["contents"] = camel_contents
        if "generationConfig" not in payload:
            payload["generationConfig"] = {}
        gc = payload["generationConfig"]
        if request.temperature is not None: gc["temperature"] = request.temperature
        if request.max_tokens is not None: gc["maxOutputTokens"] = request.max_tokens
        if request.top_p is not None: gc["topP"] = request.top_p
        if request.stop is not None: gc["stopSequences"] = request.stop
        if "gemini-3" in model_name or "gemini-2.5" in model_name:
            gc["thinkingConfig"] = {"includeThoughts": True, "thinkingLevel": "MEDIUM"}
        else:
            gc.pop("thinkingConfig", None)
        if system_texts:
            payload["systemInstruction"] = {"parts": [{"text": "\n".join(system_texts)}]}

    return payload


# ========== 认证方式解析 ==========

def _get_cookie_string() -> str:
    """获取当前可用的 Cookie 字符串（环境变量 > 大盘粘贴 > 空）"""
    return app_config.GOOGLE_COOKIE or app_state.get_google_cookie() or ""


def _resolve_auth():
    """
    解析当前可用的认证方式
    
    Returns:
        (mode, headers, url_or_none)
        mode: "cookie_direct" | "captured" | None
    """
    cookie_str = _get_cookie_string()

    # 优先：Cookie 直连模式
    if cookie_str and extract_sapisid(cookie_str):
        headers = build_auth_headers(cookie_str)
        if headers:
            return "cookie_direct", headers, None

    # 其次：浏览器捕获的凭证
    auth_bundle = app_state.get_auth_bundle()
    if auth_bundle and auth_bundle.get("headers"):
        raw_headers = {k.lower(): str(v) for k, v in auth_bundle["headers"].items()}
        raw_headers.pop("accept-encoding", None)
        raw_headers.pop("content-length", None)
        raw_headers.pop("host", None)
        raw_headers.pop("connection", None)
        raw_headers["content-type"] = "application/json"
        raw_headers["referer"] = "https://console.cloud.google.com/"
        raw_headers["origin"] = "https://console.cloud.google.com"
        return "captured", raw_headers, auth_bundle.get("url")

    return None, None, None


class HeadlessProxyUpstream(BaseUpstream):
    """
    代理通道 - 支持 Cookie 直连 和 浏览器凭证捕获 双模式
    """
    async def chat_completions(self, request_obj: OpenAIRequest, fastapi_request: Request):
        mode, headers, captured_url = _resolve_auth()

        if mode is None:
            return JSONResponse(
                status_code=401,
                content={"error": {"message": (
                    "Studio 代理凭证尚未就绪。\n"
                    "请在 Render 环境变量中设置 GOOGLE_COOKIE 和 GOOGLE_PROJECT_ID，\n"
                    "或在大盘控制台中粘贴 Cookie。"
                ), "type": "auth_error"}}
            )

        base_model_name = request_obj.model
        if base_model_name.endswith("-search"):
            base_model_name = base_model_name[:-len("-search")]

        # 确定 URL 和 Payload
        if mode == "cookie_direct":
            project_id = app_config.GOOGLE_PROJECT_ID or app_state.get_project_id()
            region = app_config.GOOGLE_REGION

            if not project_id:
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": (
                        "Cookie 直连模式需要配置 GOOGLE_PROJECT_ID。\n"
                        "请在 Render 环境变量中设置，或在大盘中填写。\n"
                        "可从 Google Cloud Console URL 中获取：console.cloud.google.com/vertex-ai?project=YOUR_PROJECT_ID"
                    ), "type": "config_error"}}
                )

            url = build_vertex_url(project_id, region, base_model_name, stream=request_obj.stream)
            payload = _build_fresh_payload(base_model_name, request_obj)
        else:
            # 捕获模式
            url = captured_url
            auth_bundle = app_state.get_auth_bundle()
            payload = _build_payload_from_template(base_model_name, request_obj, auth_bundle.get("body", {}))

        # 客户端网络配置
        client_kwargs = {"timeout": 120.0, "follow_redirects": True}
        if app_config.PROXY_URL:
            client_kwargs["proxy"] = app_config.PROXY_URL
        if app_config.SSL_CERT_FILE:
            client_kwargs["verify"] = app_config.SSL_CERT_FILE

        is_stream = request_obj.stream

        # ========== 流式处理 ==========
        if is_stream:
            async def stream_generator():
                response_id = f"chatcmpl-studio-{int(time.time())}"

                try:
                    # Cookie 直连模式每次请求重新计算 SAPISIDHASH
                    req_headers = headers
                    if mode == "cookie_direct":
                        req_headers = build_auth_headers(_get_cookie_string()) or headers

                    async with httpx.AsyncClient(**client_kwargs) as client:
                        async with client.stream("POST", url, headers=req_headers, json=payload) as response:
                            if response.status_code != 200:
                                error_text = await response.aread()
                                error_msg = error_text.decode('utf-8', errors='replace')
                                yield f"data: {json.dumps({'error': f'Studio Error {response.status_code}: {error_msg[:500]}'})}\\n\\n"
                                return

                            buffer = ""
                            async for chunk in response.aiter_text():
                                if not chunk:
                                    continue
                                buffer += chunk

                                while True:
                                    start_idx = buffer.find('{')
                                    if start_idx == -1:
                                        buffer = ""
                                        break

                                    brace_count = 0
                                    in_string = False
                                    escape = False
                                    end_idx = -1

                                    for i in range(start_idx, len(buffer)):
                                        char = buffer[i]
                                        if escape:
                                            escape = False
                                            continue
                                        if char == '\\\\':
                                            escape = True
                                            continue
                                        if char == '"':
                                            in_string = not in_string
                                            continue
                                        if not in_string:
                                            if char == '{':
                                                brace_count += 1
                                            elif char == '}':
                                                brace_count -= 1
                                                if brace_count == 0:
                                                    end_idx = i
                                                    break

                                    if end_idx != -1:
                                        json_str = buffer[start_idx:end_idx + 1]
                                        buffer = buffer[end_idx + 1:]
                                        try:
                                            obj = json.loads(json_str)
                                            gemini_chunk = types.GenerateContentResponse(**obj)
                                            yield convert_chunk_to_openai(gemini_chunk, request_obj.model, response_id, 0)
                                        except Exception:
                                            pass
                                    else:
                                        buffer = buffer[start_idx:]
                                        break

                            yield "data: [DONE]\\n\\n"

                except Exception as e:
                    print(f"❌ [StudioProxy] 流式异常: {e}")
                    traceback.print_exc()
                    yield f"data: {json.dumps({'error': f'Stream error: {str(e)}'})}\\n\\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        # ========== 非流式处理 ==========
        else:
            try:
                req_headers = headers
                if mode == "cookie_direct":
                    req_headers = build_auth_headers(_get_cookie_string()) or headers

                async with httpx.AsyncClient(**client_kwargs) as client:
                    response = await client.post(url, headers=req_headers, json=payload)
                    if response.status_code != 200:
                        return JSONResponse(status_code=response.status_code, content={"error": response.text})

                    obj = response.json()
                    gemini_response = types.GenerateContentResponse(**obj)
                    from message_processing import convert_to_openai_format
                    openai_response = convert_to_openai_format(gemini_response, request_obj.model)
                    return JSONResponse(content=openai_response)
            except Exception as e:
                print(f"❌ [StudioProxy] 非流式异常: {e}")
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": f"Studio proxy error: {str(e)}"})
