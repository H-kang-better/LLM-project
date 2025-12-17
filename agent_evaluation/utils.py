# 工具函数模块
#
# 此模块提供了项目中使用的通用工具函数
# 主要功能：
# 1. 数据哈希计算：用于比较数据库状态是否相同（tau_bench奖励计算）
# 2. OpenAI对话生成：用于用户模拟器

from hashlib import sha256
from typing import Any, Callable, Dict, List, Type, Optional, Set, Union, Tuple

# 可哈希的类型定义
ToHashable = Union[
    str, int, float, Dict[str, "ToHashable"], List["ToHashable"], Set["ToHashable"]
]
Hashable = Union[str, int, float, Tuple["Hashable"], Tuple[Tuple[str, "Hashable"]]]

def to_hashable(item: ToHashable) -> Hashable:
    """
    将Python数据结构转换为可哈希的形式

    Python的dict、list、set不能直接哈希，需要转换为tuple

    Args:
        item: 要转换的数据（dict、list、set或基本类型）

    Returns:
        可哈希的tuple或基本类型
    """
    if isinstance(item, dict):
        # dict转换为排序后的tuple of tuples
        return tuple((key, to_hashable(value)) for key, value in sorted(item.items()))
    elif isinstance(item, list):
        # list转换为tuple
        return tuple(to_hashable(element) for element in item)
    elif isinstance(item, set):
        # set转换为排序后的tuple
        return tuple(sorted(to_hashable(element) for element in item))
    else:
        # 基本类型直接返回
        return item

def consistent_hash(
    value: Hashable,
) -> str:
    """
    计算一致性哈希值

    使用SHA256算法计算哈希，确保相同的数据总是产生相同的哈希值

    Args:
        value: 可哈希的值

    Returns:
        十六进制哈希字符串
    """
    return sha256(str(value).encode("utf-8")).hexdigest()


def generate_conversation(client, model_id, messages, system_prompt=None, max_token=8192) -> str:
    """
    使用OpenAI API生成对话响应

    此函数用于用户模拟器，通过OpenAI的Chat Completions API生成用户消息

    Args:
        client: OpenAI客户端
        model_id: 模型ID（如"gpt-4", "gpt-3.5-turbo"）
        messages: 消息历史列表（Bedrock格式）
        system_prompt: 系统提示词（可选）
        max_token: 最大token数

    Returns:
        生成的响应文本
    """
    try:
        # 转换消息格式：从Bedrock格式转为OpenAI格式
        openai_messages = []

        # 添加系统提示词
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})

        # 转换消息列表
        for msg in messages:
            role = msg["role"]
            # 从Bedrock格式提取文本内容
            content = msg["content"][0]["text"] if isinstance(msg["content"], list) else msg["content"]
            openai_messages.append({"role": role, "content": content})

        # 调用OpenAI API（流式）
        response = client.chat.completions.create(
            model=model_id,
            messages=openai_messages,
            max_tokens=max_token,
            stream=True
        )

        # 处理流式响应
        out = []
        print("\nUser simulator response: ", end="")
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="")
                out.append(content)

        print()  # 换行
        result = "".join(out)
        print(f"Generated {len(result)} characters")
        return result

    except Exception as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        import traceback
        traceback.print_exc()
        return str(e)

def get_data_hash(data) -> str:
    """
    计算数据的哈希值

    用于tau_bench奖励计算：比较智能体的数据库状态和ground_truth状态是否相同

    Args:
        data: 要哈希的数据（通常是模拟数据库的dict）

    Returns:
        数据的SHA256哈希字符串
    """
    return consistent_hash(to_hashable(data))