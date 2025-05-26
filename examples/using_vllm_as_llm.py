import os
import sys
import logging
import aiohttp
import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs

# 将上级目录加入 sys.path，方便导入本地包
sys.path.append("..")

# 设置日志
logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)
# -------------------- 配置项 --------------------

# vLLM 模型 API 地址
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL = "/home/cloudos/hua/nano-graphrag-main/model/model_scope/deepseek"  # 实际模型名称以你部署时为准

# 模型最大上下文长度限制（根据您使用的模型调整）
MAX_TOKENS = 30000  # 略小于32768，留一些余量

# 简单估算token数量的函数（中英文混合文本的粗略估计）
def estimate_tokens(text):
    # 中文字符按1.5个token计算，英文按0.25个token计算（粗略估计）
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    total_chars = len(text)
    english_chars = total_chars - chinese_chars
    return int(chinese_chars * 1.5 + english_chars * 0.25)

# 根据估算长度截断消息
def truncate_messages(messages, max_tokens=MAX_TOKENS):
    # 保留最开始的系统消息（如果有）
    system_msg = None
    other_msgs = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_msg = msg
        else:
            other_msgs.append(msg)
    
    # 估算系统消息的token数（如果有）
    system_tokens = estimate_tokens(system_msg["content"]) if system_msg else 0
    available_tokens = max_tokens - system_tokens
    
    # 保留最后一条用户消息，并确保它不会被截断太多
    last_msg = other_msgs[-1] if other_msgs else None
    if last_msg and last_msg["role"] == "user":
        other_msgs = other_msgs[:-1]
        last_msg_tokens = estimate_tokens(last_msg["content"])
        # 如果最后一条消息太长，需要截断
        if last_msg_tokens > available_tokens * 0.6:  # 允许最后一条消息占用60%的可用空间
            # 截断到可用空间的60%（保留开头部分）
            max_last_msg_chars = int(len(last_msg["content"]) * (available_tokens * 0.6 / last_msg_tokens))
            last_msg["content"] = last_msg["content"][:max_last_msg_chars] + "..."
            available_tokens -= estimate_tokens(last_msg["content"])
        else:
            available_tokens -= last_msg_tokens
    else:
        last_msg = None
    
    # 遍历历史消息，从最早的开始删除，直到总token数满足要求
    result_msgs = []
    for msg in reversed(other_msgs):  # 从最新到最旧遍历
        msg_tokens = estimate_tokens(msg["content"])
        if msg_tokens <= available_tokens:
            result_msgs.insert(0, msg)  # 插入到列表开头保持顺序
            available_tokens -= msg_tokens
        else:
            # 如果太长但仍有足够空间，可以考虑截断而不是完全丢弃
            if available_tokens > 100:  # 至少保留一些内容
                max_chars = int(len(msg["content"]) * (available_tokens / msg_tokens))
                msg["content"] = msg["content"][:max_chars] + "..."
                result_msgs.insert(0, msg)
            break
    
    # 重新组合消息
    final_msgs = []
    if system_msg:
        final_msgs.append(system_msg)
    final_msgs.extend(result_msgs)
    if last_msg:
        final_msgs.append(last_msg)
    
    return final_msgs

# -------------------- vLLM LLM 异步调用函数（带缓存） --------------------
async def vllm_model_if_cache(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    """
    使用 vLLM 提供的 OpenAI API 接口进行推理，支持 system prompt 和缓存。
    """
    # 获取缓存
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # 截断消息以符合模型最大上下文限制
    messages = truncate_messages(messages, MAX_TOKENS)

    # 计算缓存键
    if hashing_kv is not None:
        args_hash = compute_args_hash(VLLM_MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    # 调用 vLLM 接口
    try:
        async with aiohttp.ClientSession() as session:
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": VLLM_MODEL,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),  # 允许外部传入温度参数
                "max_tokens": kwargs.get("max_tokens", 2048),   # 允许控制输出长度
            }
            async with session.post(
                f"{VLLM_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                timeout=3000,
                ssl=False  # 关闭 SSL 证书验证
            ) as resp:
                resp.raise_for_status()  # 若状态码非 2xx，则抛出异常
                response = await resp.json()
                result = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not result:
                    logging.error("vLLM 接口返回内容无效: %s", response)
                    raise ValueError("无效的 vLLM 接口返回")
    except Exception as e:
        logging.error("调用 vLLM 接口失败: %s", e)
        raise

    # 缓存写入
    if hashing_kv is not None:
        try:
            await hashing_kv.upsert({args_hash: {"return": result, "model": VLLM_MODEL}})
        except Exception as e:
            logging.warning("缓存写入失败: %s", e)

    return result
