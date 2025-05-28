import os
import sys
import logging
import aiohttp
from typing import List, Dict, Any, Optional, Coroutine

from nano_graphrag import GraphRAG, QueryParam # GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs

sys.path.append("..")

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# -------------------- Configuration Constants --------------------

VLLM_API_ENDPOINT = "http://localhost:8000/v1"
VLLM_MODEL_IDENTIFIER = "/home/cloudos/hua/nano-graphrag-main/model/model_scope/deepseek"

MODEL_CONTEXT_WINDOW_LIMIT = 30000

# Token estimation constants
CHINESE_CHAR_TOKEN_MULTIPLIER = 1.5
ENGLISH_CHAR_TOKEN_MULTIPLIER = 0.25

# Message truncation constants
LAST_MESSAGE_TOKEN_BUDGET_RATIO = 0.6
MIN_TOKENS_FOR_PARTIAL_MESSAGE_TRUNCATION = 100
DEFAULT_MAX_OUTPUT_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7
API_TIMEOUT_SECONDS = 3000 # Default: 3000 seconds (50 minutes)

# -------------------- Utility Functions --------------------

def approximate_token_count(text: str) -> int:
    num_chinese_chars = sum(1 for char_code in text if '\u4e00' <= char_code <= '\u9fff')
    total_character_count = len(text)
    num_english_chars = total_character_count - num_chinese_chars
    return int(num_chinese_chars * CHINESE_CHAR_TOKEN_MULTIPLIER + \
               num_english_chars * ENGLISH_CHAR_TOKEN_MULTIPLIER)

def trim_message_history_to_token_limit(
    message_list: List[Dict[str, str]],
    token_limit: int = MODEL_CONTEXT_WINDOW_LIMIT
) -> List[Dict[str, str]]:
    system_message_obj: Optional[Dict[str, str]] = None
    operational_messages: List[Dict[str, str]] = []

    for message_data in message_list:
        if message_data.get("role") == "system":
            system_message_obj = message_data
        else:
            operational_messages.append(message_data)

    system_message_tokens = approximate_token_count(system_message_obj["content"]) if system_message_obj else 0
    remaining_token_budget = token_limit - system_message_tokens

    last_user_message_obj: Optional[Dict[str, str]] = None
    if operational_messages and operational_messages[-1].get("role") == "user":
        last_user_message_obj = operational_messages.pop()
        last_user_message_tokens = approximate_token_count(last_user_message_obj["content"])

        max_tokens_for_last_message = remaining_token_budget * LAST_MESSAGE_TOKEN_BUDGET_RATIO
        if last_user_message_tokens > max_tokens_for_last_message:
            chars_to_keep_for_last_message = int(
                len(last_user_message_obj["content"]) * (max_tokens_for_last_message / last_user_message_tokens)
            )
            last_user_message_obj["content"] = last_user_message_obj["content"][:chars_to_keep_for_last_message] + "..."
            remaining_token_budget -= approximate_token_count(last_user_message_obj["content"])
        else:
            remaining_token_budget -= last_user_message_tokens
    
    trimmed_earlier_messages: List[Dict[str, str]] = []
    for message_data in reversed(operational_messages):
        current_message_tokens = approximate_token_count(message_data["content"])
        if current_message_tokens <= remaining_token_budget:
            trimmed_earlier_messages.insert(0, message_data)
            remaining_token_budget -= current_message_tokens
        else:
            if remaining_token_budget > MIN_TOKENS_FOR_PARTIAL_MESSAGE_TRUNCATION:
                chars_to_keep_for_current_message = int(
                    len(message_data["content"]) * (remaining_token_budget / current_message_tokens)
                )
                message_data["content"] = message_data["content"][:chars_to_keep_for_current_message] + "..."
                trimmed_earlier_messages.insert(0, message_data)
            break 
    
    final_message_payload: List[Dict[str, str]] = []
    if system_message_obj:
        final_message_payload.append(system_message_obj)
    final_message_payload.extend(trimmed_earlier_messages)
    if last_user_message_obj:
        final_message_payload.append(last_user_message_obj)

    return final_message_payload

# -------------------- Asynchronous vLLM API Call Function --------------------

async def fetch_vllm_completion_with_cache_async(
    user_prompt_content: str,
    optional_system_prompt_content: Optional[str] = None,
    prior_conversation_messages: Optional[List[Dict[str, str]]] = None,
    **kwargs: Any
) -> str:
    cache_storage_instance: Optional[BaseKVStorage] = kwargs.pop("hashing_kv", None)
    
    api_messages_payload: List[Dict[str, str]] = []
    if optional_system_prompt_content:
        api_messages_payload.append({"role": "system", "content": optional_system_prompt_content})
    if prior_conversation_messages:
        api_messages_payload.extend(prior_conversation_messages)
    api_messages_payload.append({"role": "user", "content": user_prompt_content})

    api_messages_payload = trim_message_history_to_token_limit(api_messages_payload, MODEL_CONTEXT_WINDOW_LIMIT)

    cache_key: Optional[str] = None
    if cache_storage_instance:
        cache_key = compute_args_hash(VLLM_MODEL_IDENTIFIER, api_messages_payload)
        cached_response_data = await cache_storage_instance.get_by_id(cache_key)
        if cached_response_data is not None and "return" in cached_response_data:
            return str(cached_response_data["return"])

    llm_generated_content: str = ""
    api_request_payload = {
        "model": VLLM_MODEL_IDENTIFIER,
        "messages": api_messages_payload,
        "temperature": kwargs.get("temperature", DEFAULT_TEMPERATURE),
        "max_tokens": kwargs.get("max_tokens", DEFAULT_MAX_OUTPUT_TOKENS),
    }

    try:
        async with aiohttp.ClientSession() as http_session:
            request_headers = {"Content-Type": "application/json"}
            client_timeout = aiohttp.ClientTimeout(total=API_TIMEOUT_SECONDS)
            
            async with http_session.post(
                f"{VLLM_API_ENDPOINT}/chat/completions",
                json=api_request_payload,
                headers=request_headers,
                timeout=client_timeout,
                ssl=False 
            ) as http_response:
                http_response.raise_for_status()
                json_response_body = await http_response.json()
                
                choices = json_response_body.get("choices")
                if choices and isinstance(choices, list) and choices[0]:
                    message_content = choices[0].get("message", {}).get("content")
                    if message_content:
                        llm_generated_content = str(message_content)
                    else:
                        logging.error("vLLM API response 'content' is missing or empty: %s", json_response_body)
                        raise ValueError("Invalid vLLM API response: 'content' field missing or empty.")
                else:
                    logging.error("vLLM API response 'choices' are invalid or missing: %s", json_response_body)
                    raise ValueError("Invalid vLLM API response: 'choices' field malformed.")
    
    except aiohttp.ClientError as client_err:
        logging.error("AIOHTTP client error during vLLM API call: %s", client_err)
        raise
    except ValueError as val_err:
        logging.error("Data error processing vLLM API response: %s", val_err)
        raise
    except Exception as e:
        logging.error("Unexpected error calling vLLM API: %s", e)
        raise

    if cache_storage_instance and cache_key:
        try:
            await cache_storage_instance.upsert({cache_key: {"return": llm_generated_content, "model": VLLM_MODEL_IDENTIFIER}})
        except Exception as e:
            logging.warning("Failed to write to cache: %s", e)

    return llm_generated_content
