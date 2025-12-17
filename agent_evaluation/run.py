# LangChain版本的运行模块
#
# 此模块实现了基于LangChain框架的tau_bench智能体评估
# 主要功能：
# 1. 创建LangChain LLM实例（支持OpenAI、DashScope、LiteLLM等）
# 2. 运行tau_bench任务评估循环
# 3. 收集和保存评估结果
# 4. 计算评估指标（平均奖励、Pass^k等）

import json
import os
import time
import random
import traceback
from typing import List
from datetime import datetime
from math import comb
import multiprocessing
from dotenv import load_dotenv

# tau_bench类型导入
from tau_bench.types import EnvRunResult, RunConfig

# LangChain imports
from langchain_community.chat_models import ChatLiteLLM
from langchain_openai import ChatOpenAI

from env import EnvLangChain
from data import load_data


# 修改JSON序列化以正确显示中文字符
original_dumps = json.dumps
def custom_dumps(*args, **kwargs):
    """自定义JSON序列化函数，确保中文正确显示"""
    kwargs['ensure_ascii'] = False
    return original_dumps(*args, **kwargs)
json.dumps = custom_dumps


load_dotenv(".env", override=True)

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "agent_evaluation"

# 从环境变量加载API配置
API_KEY = os.getenv("DASHSCOPE_API_KEY") if os.getenv("DASHSCOPE_API_KEY") else os.environ.get('LLM_API_KEY')
API_URL = os.environ.get("LLM_BASE_URL")

# Langfuse追踪配置（可选）
# Langfuse是一个LLM应用的可观测性平台
public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
langfuse_endpoint = os.environ.get("LANGFUSE_HOST")

# 如果配置了Langfuse，设置追踪回调处理器
if public_key and secret_key and langfuse_endpoint:
    from langfuse import CallbackHandler
    langfuse_handler = CallbackHandler(
        public_key=public_key,
        secret_key=secret_key,
        host=langfuse_endpoint
    )
else:
    langfuse_handler = None


def create_llm(config: RunConfig):
    """
    根据tau_bench RunConfig创建LangChain LLM实例

    支持多种模型提供商：
    - OpenAI及兼容API（使用ChatOpenAI）
    - DashScope、Anthropic等（使用ChatLiteLLM）

    Args:
        config: tau_bench的RunConfig对象，包含模型配置

    Returns:
        LangChain LLM实例
    """
    if config.model_provider == "openai":
        # OpenAI或兼容的API
        llm = ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            base_url=API_URL,
            api_key=API_KEY,
        )
    elif config.model_provider in ["dashscope", "anthropic", "litellm"] or True:
        # 使用LiteLLM支持其他提供商
        llm = ChatLiteLLM(
            model=config.model,
            temperature=config.temperature,
            api_base=API_URL,
            api_key=API_KEY,
        )
    else:
        raise ValueError(f"Unsupported model provider: {config.model_provider}")

    return llm


def run(config: RunConfig) -> List[EnvRunResult]:
    """
    运行基于LangChain的tau_bench智能体评估

    流程：
    1. 根据config加载tau_bench任务列表（test/train/dev）
    2. 创建LangChain LLM实例
    3. 对每个任务创建EnvLangChain环境并运行
    4. 收集tau_bench标准的EnvRunResult结果
    5. 计算并显示评估指标
    6. 保存结果到JSON文件

    Args:
        config: tau_bench的RunConfig配置对象

    Returns:
        List[EnvRunResult]: tau_bench标准格式的评估结果列表
    """
    # 验证tau_bench支持的环境类型
    assert config.env in ["retail", "airline"], "Only retail and airline envs are supported"
    assert config.task_split in ["train", "test", "dev"], "Invalid task split"

    random.seed(config.seed)
    time_str = datetime.now().strftime("%m%d%H%M%S")
    # 构建结果文件路径
    ckpt_path = f"{config.log_dir}/langchain-{config.agent_strategy}-{config.model.split('/')[-1]}-{config.temperature}_range_{config.start_index}-{config.end_index}_user-{config.user_model.split('/')[-1]}-{config.user_strategy}_{time_str}.json"

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    # 加载系统提示词（包含环境知识库）
    with open(os.path.join("./", "wiki.md"), "r") as f:
        system_prompt = f.read()

    # 根据任务集合类型从tau_bench加载任务
    match config.task_split:
        case "test":
            from tau_bench.envs.retail.tasks_test import TASKS_TEST as tasks
        case "train":
            from tau_bench.envs.retail.tasks_train import TASKS_TRAIN as tasks
        case "dev":
            from tau_bench.envs.retail.tasks_dev import TASKS_DEV as tasks
        case _:
            raise ValueError(f"Unknown task split: {config.task_split}")

    end_index = (
        len(tasks) if config.end_index == -1 else min(config.end_index, len(tasks))
    )

    results: List[EnvRunResult] = []
    lock = multiprocessing.Lock()

    if config.task_ids and len(config.task_ids) > 0:
        print(f"Running tasks {config.task_ids} (checkpoint path: {ckpt_path})")
    else:
        print(
            f"Running tasks {config.start_index} to {end_index} (checkpoint path: {ckpt_path})"
        )

    # 创建LangChain LLM实例
    llm = create_llm(config)

    total_cost = 0

    # 执行多轮试验
    for i in range(config.num_trials):
        if config.task_ids and len(config.task_ids) > 0:
            idxs = config.task_ids
        else:
            idxs = list(range(config.start_index, end_index))

        if config.shuffle:
            random.shuffle(idxs)

        def _run(idx: int, total_cost: float) -> tuple[EnvRunResult, float]:
            """
            运行单个tau_bench任务

            Args:
                idx: tau_bench任务索引
                total_cost: 累计成本

            Returns:
                (EnvRunResult, 更新后的成本)
            """
            print(f"Running task {idx}")
            try:
                # 创建LangChain环境，使用tau_bench任务列表
                env = EnvLangChain(
                    tasks=tasks,  # tau_bench任务列表
                    llm=llm,
                    system_prompt=system_prompt,
                    terminate_tools=["transfer_to_human_agents"],
                    task_index=idx,
                    config=config
                )

                # 运行环境循环（智能体-用户模拟器交互）
                res = env.loop()
                total_cost += res.total_cost

                # 构建tau_bench标准的EnvRunResult
                result = EnvRunResult(
                    task_id=idx,
                    reward=res.reward,
                    info=res.info,
                    traj=res.messages,
                    trial=i,
                )

            except Exception as e:
                # 捕获异常并记录
                result = EnvRunResult(
                    task_id=idx,
                    reward=0.0,
                    info={"error": str(e), "traceback": traceback.format_exc()},
                    traj=[],
                    trial=i,
                )

            print(
                "✅" if result.reward == 1 else "❌",
                f"task_id={idx}",
                result.info,
            )
            print("-----")

            # 保存检查点
            with lock:
                data = []
                if os.path.exists(ckpt_path):
                    with open(ckpt_path, "r") as f:
                        data = json.load(f)
                with open(ckpt_path, "w") as f:
                    json.dump(data + [result.model_dump()], f, indent=2)

            return result, total_cost

        # 顺序运行任务（可修改为并行执行）
        for idx in idxs:
            result, total_cost = _run(idx, total_cost)
            results.append(result)
            time.sleep(5)  # 速率限制

        print(f"Total cost: ${total_cost:.4f}")

    # 显示评估指标
    display_metrics(results, config.num_trials)

    # 保存最终结果
    with open(ckpt_path, "w") as f:
        json.dump([result.model_dump() for result in results], f, indent=2)
        print(f"\n📄 Results saved to {ckpt_path}\n")

    return results


def display_metrics(results: List[EnvRunResult], num_trials) -> None:
    """
    显示tau_bench评估指标

    计算并显示：
    1. 平均奖励（Average Reward）
    2. Pass^k指标：在k次尝试中至少成功一次的概率

    Pass^k是代码生成和智能体评估中常用的指标
    """
    def is_successful(reward: float) -> bool:
        """判断任务是否成功（reward接近1.0）"""
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    rewards = [r.reward for r in results]
    avg_reward = sum(rewards) / len(rewards)

    # 计算每个任务的成功次数
    c_per_task_id: dict[int, int] = {}
    for result in results:
        if result.task_id not in c_per_task_id:
            c_per_task_id[result.task_id] = 1 if is_successful(result.reward) else 0
        else:
            c_per_task_id[result.task_id] += 1 if is_successful(result.reward) else 0

    # 计算Pass^k指标
    # Pass^k = 平均每个任务在k次尝试中至少成功一次的概率
    pass_hat_ks: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k = 0
        for c in c_per_task_id.values():
            # 使用组合数学计算：从n次尝试中选k次，至少有一次成功的概率
            sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)

    print(f"🏆 Average reward: {avg_reward}")
    print("📈 Pass^k")
    for k, pass_hat_k in pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")