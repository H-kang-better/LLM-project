# LangChain版本的主入口文件
#
# 此模块为LangChain版本的tau_bench评估提供命令行接口
# 功能：
# 1. 解析命令行参数（模型、环境、任务配置等）
# 2. 构建tau_bench的RunConfig配置对象
# 3. 调用run_langchain.run()执行评估

import argparse
from tau_bench.types import RunConfig  # tau_bench的配置类型
from run import run
from litellm import provider_list  # LiteLLM支持的提供商列表
from tau_bench.envs.user import UserStrategy  # tau_bench的用户模拟器策略


def parse_args() -> RunConfig:
    """
    解析命令行参数并构建tau_bench RunConfig

    支持的参数包括：
    - 模型配置：model, model_provider, temperature
    - 用户模拟器配置：user_model, user_model_provider, user_strategy
    - 任务配置：env, task_split, start_index, end_index, task_ids
    - 运行配置：num_trials, seed, shuffle, max_concurrency
    - 输出配置：log_dir

    Returns:
        RunConfig: tau_bench标准的运行配置对象
    """
    parser = argparse.ArgumentParser(
        description="使用LangChain框架运行智能体评估"
    )
    parser.add_argument("--num-trials", type=int, default=1,
                        help="每个任务的试验次数")
    parser.add_argument(
        "--env", type=str, choices=["retail", "airline"], default="retail",
        help="要使用的tau_bench环境类型"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen-turbo",
        help="智能体使用的模型（例如：'gpt-4', 'claude-3-opus-20240229'）"
    )
    parser.add_argument(
        "--model-provider",
        type=str,
        choices=provider_list + ["bedrock", "openai"],  # LiteLLM提供商列表 + 自定义提供商
        default="openai",
        help="智能体的模型提供商"
    )
    parser.add_argument(
        "--user-model",
        type=str,
        default="qwen-turbo",
        help="用户模拟器使用的模型"
    )
    parser.add_argument(
        "--user-model-provider",
        type=str,
        choices=provider_list + ["bedrock", "openai"],
        default="openai",
        help="用户模拟器的模型提供商"
    )
    parser.add_argument(
        "--agent-strategy",
        type=str,
        default="tool-calling",
        choices=["tool-calling", "act", "react", "few-shot"],
        help="智能体策略（LangChain默认使用tool-calling）"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="智能体模型的采样温度"
    )
    parser.add_argument(
        "--task-split",
        type=str,
        default="test",
        choices=["train", "test", "dev"],
        help="要运行的tau_bench任务集合"
    )
    parser.add_argument("--start-index", type=int, default=0,
                        help="要运行的任务起始索引")
    parser.add_argument("--end-index", type=int, default=-1,
                        help="要运行的任务结束索引（-1表示全部）")
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0],
                        help="（可选）仅运行指定ID的任务")
    parser.add_argument("--log-dir", type=str, default="results",
                        help="保存结果的目录")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="并行运行的任务数（当前仅支持1）"
    )
    parser.add_argument("--seed", type=int, default=10,
                        help="随机种子")
    parser.add_argument("--shuffle", type=int, default=0,
                        help="是否打乱任务顺序（1表示是，0表示否）")
    parser.add_argument("--user-strategy", type=str, default="llm",
                        choices=[item.value for item in UserStrategy],  # tau_bench的用户策略枚举
                        help="用户模拟器策略")
    parser.add_argument("--few-shot-displays-path", type=str,
                        help="包含few-shot示例的jsonlines文件路径")

    args = parser.parse_args()
    # 打印配置信息
    print("=" * 80)
    print("LangChain智能体评估配置")
    print("=" * 80)
    print(f"模型: {args.model}")
    print(f"模型提供商: {args.model_provider}")
    print(f"用户模型: {args.user_model}")
    print(f"环境: {args.env}")
    print(f"任务集合: {args.task_split}")
    print(f"温度: {args.temperature}")
    print(f"智能体策略: {args.agent_strategy}")
    print("=" * 80)

    # 构建tau_bench标准的RunConfig对象
    return RunConfig(
        model_provider=args.model_provider,
        user_model_provider=args.user_model_provider,
        model=args.model,
        user_model=args.user_model,
        num_trials=args.num_trials,
        env=args.env,
        agent_strategy=args.agent_strategy,
        temperature=args.temperature,
        task_split=args.task_split,
        start_index=args.start_index,
        end_index=args.end_index,
        task_ids=args.task_ids,
        log_dir=args.log_dir,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        shuffle=args.shuffle,
        user_strategy=args.user_strategy,
        few_shot_displays_path=args.few_shot_displays_path,
    )


def main():
    """
    主入口函数

    解析命令行参数并运行tau_bench评估
    """
    config = parse_args()
    run(config)



if __name__ == "__main__":
    main()