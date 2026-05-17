import asyncio
import json
import re
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Google specific imports
from google.genai import types
from google import genai

# Local module imports
from models import OpenAIRequest
from auth import get_api_key
from message_processing import (
    create_gemini_prompt,
)
from api_helpers import (
    create_generation_config,
    create_openai_error_response,
    execute_gemini_call,
)
from openai_handler import OpenAIDirectHandler
from project_id_discovery import discover_project_id

router = APIRouter()

@router.post("/v1/chat/completions")
async def chat_completions(fastapi_request: Request, request: OpenAIRequest, api_key: str = Depends(get_api_key)):
    try:
        credential_manager_instance = fastapi_request.app.state.credential_manager
        express_key_manager_instance = fastapi_request.app.state.express_key_manager
        
        OPENAI_DIRECT_SUFFIX = "-openai"
        OPENAI_SEARCH_SUFFIX = "-openaisearch"
        EXPERIMENTAL_MARKER = "-exp-"
        PAY_PREFIX = "[PAY]"
        EXPRESS_PREFIX = "[EXPRESS] " 
        
        base_model_name = request.model 
        
        is_express_model_request = False
        if base_model_name.startswith(EXPRESS_PREFIX):
            is_express_model_request = True
            base_model_name = base_model_name[len(EXPRESS_PREFIX):]

        if base_model_name.startswith(PAY_PREFIX):
            base_model_name = base_model_name[len(PAY_PREFIX):]

        is_openai_direct_model = False
        is_openai_search_model = False
        
        if base_model_name.endswith(OPENAI_SEARCH_SUFFIX):
            is_openai_search_model = True
            is_openai_direct_model = True
            base_model_name = base_model_name[:-len(OPENAI_SEARCH_SUFFIX)]
        elif base_model_name.endswith(OPENAI_DIRECT_SUFFIX):
            is_openai_direct_model = True
            base_model_name = base_model_name[:-len(OPENAI_DIRECT_SUFFIX)]
            
        if EXPERIMENTAL_MARKER in base_model_name:
            is_openai_direct_model = True

        is_grounded_search = base_model_name.endswith("-search")
        if is_grounded_search: base_model_name = base_model_name[:-len("-search")]

        # ==========================================
        # 核心：智能识别 image 并配置 (Gemini 3 生图增强)
        # ==========================================
        is_image_model = "image" in request.model.lower()
        if is_image_model:
            is_openai_direct_model = False
            
        gen_config_dict = create_generation_config(request)

        is_thinking_capable = "gemini-2.5" in base_model_name or "gemini-3" in base_model_name

        if is_thinking_capable:
            if "thinking_config" not in gen_config_dict:
                gen_config_dict["thinking_config"] = {}
            gen_config_dict["thinking_config"]["include_thoughts"] = True

        # ==========================================
        # 🎨 生图模型通用超清与联网增强逻辑
        # ==========================================
        if is_image_model:
            print(f"🎨 [生图增强模式] 激活！目标模型: {base_model_name}")
            
            # 1. 强制开启深度思考，拉大预算 (强迫模型生图前必须构思)
            gen_config_dict["thinking_config"]["include_thoughts"] = True
            
            # 2. 强制挂载 Google Search 工具 (解决实体认知错误、画错角色的痛点)
            search_tool = types.Tool(google_search=types.GoogleSearch())
            if "tools" in gen_config_dict and isinstance(gen_config_dict["tools"], list):
                if not any(hasattr(t, "google_search") for t in gen_config_dict["tools"]):
                    gen_config_dict["tools"].append(search_tool)
            else:
                gen_config_dict["tools"] = [search_tool]

            # 3. 动态嗅探前端分辨率与比例 (支持 size 参数和 --ar 指令)
            target_aspect_ratio = "1:1" # 默认比例
            extra_params = getattr(request, "model_extra", {}) or {}
            size_param = extra_params.get("size")
            if size_param:
                if size_param == "1024x1024": target_aspect_ratio = "1:1"
                elif size_param == "1024x768": target_aspect_ratio = "4:3"
                elif size_param == "768x1024": target_aspect_ratio = "3:4"
                elif size_param in ["1:1", "9:16", "16:9", "3:4", "4:3"]:
                    target_aspect_ratio = size_param
            
            # 提取最后一条用户提问中的 --ar 比例
            last_user_msg = ""
            for msg in reversed(request.messages):
                if msg.role == "user":
                    if isinstance(msg.content, str):
                        last_user_msg = msg.content
                    elif isinstance(msg.content, list):
                        last_user_msg = " ".join([p.get("text", "") for p in msg.content if isinstance(p, dict) and p.get("type") == "text"])
                    break
                    
            ar_match = re.search(r'(?i)(?:--ar\s+)?(1[:：]1|16[:：]9|9[:：]16|3[:：]4|4[:：]3)', last_user_msg)
            if ar_match:
                target_aspect_ratio = ar_match.group(1).replace("：", ":")
            
            # 4. 注入高质量通用 System Prompt (绝不限定题材，只要求画质、比例和事实核查)
            image_sys_prompt = (
                f"You are an expert AI image generation and reasoning assistant. "
                f"Your task is to generate the requested image with absolute ultra-high visual quality, targeting 4K-level resolution detail, strictly avoiding any compression artifacts, blurring, or logical distortion. "
                f"STRICTLY adhere to the requested output aspect ratio: {target_aspect_ratio}. "
                f"CRITICAL INSTRUCTION: Before generating the final image, YOU MUST deeply think about the composition. "
                f"If the user requests ANY specific entity (such as a character, movie, game, realistic person, specific object, or historical style) that you are not 100% certain about, "
                f"YOU MUST use your internal Google Search tool to search for their exact visual appearance and features first. Base your image accurately on those search results."
            )
            existing_sys = gen_config_dict.get("system_instruction", "")
            gen_config_dict["system_instruction"] = f"{image_sys_prompt}\n\n{existing_sys}".strip()
            print(f"🎨 [生图参数组装] 目标比例: {target_aspect_ratio} | 联网搜索: 已挂载 | 超清系统指令: 已注入")

        # ==========================================

        client_to_use = None

        if is_express_model_request:
            if express_key_manager_instance.get_total_keys() == 0:
                error_msg = f"Model '{request.model}' requires an Express API key, but none are configured."
                return JSONResponse(status_code=401, content=create_openai_error_response(401, error_msg, "authentication_error"))

            total_keys = express_key_manager_instance.get_total_keys()
            for attempt in range(total_keys):
                key_tuple = express_key_manager_instance.get_express_api_key()
                if key_tuple:
                    original_idx, key_val = key_tuple
                    try:
                        if "gemini-2.5-pro" in base_model_name or "gemini-2.5-flash" in base_model_name:
                            project_id = await discover_project_id(key_val)
                            base_url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global"
                            client_to_use = genai.Client(
                                vertexai=True,
                                api_key=key_val,
                                http_options=types.HttpOptions(base_url=base_url)
                            )
                            client_to_use._api_client._http_options.api_version = None
                        else:
                            client_to_use = genai.Client(vertexai=True, api_key=key_val)
                        break 
                    except Exception as e:
                        client_to_use = None 
                else:
                    client_to_use = None

            if client_to_use is None: 
                return JSONResponse(status_code=500, content=create_openai_error_response(500, "All configured Express API keys failed.", "server_error"))
        
        else: 
            rotated_credentials, rotated_project_id = credential_manager_instance.get_credentials()
            
            if rotated_credentials and rotated_project_id:
                try:
                    client_to_use = genai.Client(vertexai=True, credentials=rotated_credentials, project=rotated_project_id, location="global")
                except Exception as e:
                    return JSONResponse(status_code=500, content=create_openai_error_response(500, str(e), "server_error"))
            else: 
                return JSONResponse(status_code=401, content=create_openai_error_response(401, "No SA credentials available.", "authentication_error"))

        if not is_openai_direct_model and client_to_use is None:
            return JSONResponse(status_code=500, content=create_openai_error_response(500, "Critical internal server error: Gemini client not initialized.", "server_error"))

        if is_openai_direct_model:
            if is_express_model_request:
                openai_handler = OpenAIDirectHandler(express_key_manager=express_key_manager_instance)
                return await openai_handler.process_request(request, base_model_name, is_express=True, is_openai_search=is_openai_search_model)
            else:
                openai_handler = OpenAIDirectHandler(credential_manager=credential_manager_instance)
                return await openai_handler.process_request(request, base_model_name, is_openai_search=is_openai_search_model)
        else: 
            current_prompt_func = create_gemini_prompt

            if is_grounded_search and not is_image_model:
                search_tool = types.Tool(google_search=types.GoogleSearch())
                if "tools" in gen_config_dict and isinstance(gen_config_dict["tools"], list):
                    gen_config_dict["tools"].append(search_tool)
                else:
                    gen_config_dict["tools"] = [search_tool]

            return await execute_gemini_call(client_to_use, base_model_name, current_prompt_func, gen_config_dict, request)

    except Exception as e:
        error_msg = f"Unexpected error in chat_completions endpoint: {str(e)}"
        print(error_msg)
        return JSONResponse(status_code=500, content=create_openai_error_response(500, error_msg, "server_error"))