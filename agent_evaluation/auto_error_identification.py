# Copyright Sierra
#
# 此脚本用于自动识别和分析智能体执行失败的原因
# 主要功能：
# 1. 从tau_bench测试结果中提取失败的轨迹
# 2. 使用大语言模型分析失败原因归属（用户/智能体/环境）
# 3. 对智能体导致的失败进行类型分类（调用错误工具/参数错误/目标部分完成/其他）

import json
import argparse
from enum import Enum
from pydantic import BaseModel
# tau_bench 导入：核心工具和类型
from tau_bench.model_utils import default_api_from_args, API  # 从命令行参数创建API实例
from tau_bench.envs.airline.tasks_test import TASKS as AIRLINE_TASKS  # 航空领域测试任务
from tau_bench.envs.retail.tasks_test import TASKS_TEST as RETAIL_TASKS  # 零售领域测试任务
from tau_bench.model_utils.args import api_parser  # 命令行参数解析器
from tau_bench.types import Task, Action  # tau_bench核心类型定义
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor  # 多线程并发执行

def get_args() -> argparse.Namespace:
    """
    解析命令行参数

    使用tau_bench的api_parser作为基础，添加错误分析特定的参数
    """
    parser = api_parser()  # tau_bench提供的API参数解析器（包含模型配置等）
    parser.add_argument("--env", type=str, required=True, choices=["airline", "retail"],
                       help="环境类型：airline（航空）或retail（零售）")
    parser.add_argument("--results-path", type=str, help="tau_bench测试结果文件路径")
    parser.add_argument("--max-concurrency", type=int, default=1, help="最大并发API调用数")
    parser.add_argument("--output-path", type=str, required=True, help="分析结果输出文件路径")
    parser.add_argument("--max-num-failed-results", "-n", type=int, help="最多分析的失败结果数量")
    return parser.parse_args()

class OriginalResult(BaseModel):
    """
    原始测试结果数据模型
    包含从tau_bench结果文件中提取的任务执行信息
    """
    task_id: int  # tau_bench任务ID
    user_instruction: str  # 用户指令（从tau_bench Task中提取）
    traj: List[Dict[str, Any]]  # 对话轨迹
    ground_truth_actions: List[Action]  # tau_bench定义的标准动作序列
    ground_truth_outputs: List[str]  # tau_bench定义的期望输出

class FaultAuthor(Enum):
    """
    错误归属枚举：标识谁应该为失败负责
    """
    USER = "user"  # 用户模拟器的问题
    AGENT = "agent"  # 智能体的问题
    ENVIRONMENT = "environment"  # 环境或其他问题

class FaultAssignmentResult(BaseModel):
    """
    错误归属分析结果
    使用LLM分类器判断失败原因归属于哪一方
    """
    task_id: int
    author: FaultAuthor  # 责任方
    description: str  # LLM生成的详细描述

    def model_dump(self) -> Dict[str, Any]:
        """序列化为字典格式"""
        return {
            "task_id": self.task_id,
            "author": self.author.value,
            "description": self.description,
        }

class FaultType(Enum):
    """
    错误类型枚举：对智能体导致的失败进行细分
    """
    CALLED_WRONG_TOOL = "called_wrong_tool"  # 调用了错误的工具
    USED_WRONG_TOOL_ARGUMENT = "used_wrong_tool_argument"  # 工具参数错误
    GOAL_PARTIALLY_COMPLETED = "goal_partially_completed"  # 目标只完成了一部分
    OTHER = "other"  # 其他类型的错误

class FaultTypeResult(BaseModel):
    """
    错误类型分析结果
    仅对归属于智能体的失败进行类型分类
    """
    task_id: int
    fault_type: FaultType  # 错误类型
    description: str  # LLM生成的详细描述

    def model_dump(self) -> Dict[str, Any]:
        """序列化为字典格式"""
        return {
            "task_id": self.task_id,
            "fault_type": self.fault_type.value,
            "description": self.description,
        }

class GradingStrategy(Enum):
    """
    评分策略：根据tau_bench Task定义选择评估方式
    """
    ACTIONS = "actions"  # 基于动作序列评估
    OUTPUTS = "outputs"  # 基于输出内容评估

def context_description(grading_strategy: GradingStrategy) -> str:
    """
    生成评分策略对应的上下文描述
    用于向LLM解释输入数据的含义
    """
    if grading_strategy == GradingStrategy.ACTIONS:
        return """You will be given a user instruction, the ground truth action sequence, and a trajectory.
- The user instruction is the instruction given to the simulated user.
- The ground truth action sequence is one example of a valid sequence of actions that lead to the goal state (the sequence of actions could be empty, meaning that no action should have been taken).
- The trajectory is the sequence of messages between the user and the agent.
- The trajectory has been determined to have a fault."""
    return """You will be given a user instruction, the set of required agent response outputs, and a trajectory.
- The user instruction is the instruction given to the simulated user.
- The required agent response outputs are the set of outputs that the agent is expected to communicate to the user.
- The trajectory is the sequence of messages between the user and the agent.
- The trajectory has been determined to have a fault."""

def display_traj(traj: List[Dict[str, Any]]) -> str:
    """
    格式化对话轨迹为可读字符串
    过滤掉系统消息，只保留用户和智能体的对话
    """
    if len(traj) == 0:
        raise ValueError("Trajectory is empty")
    stripped_traj = [item for item in traj if item["role"] != "system"]
    return "\n".join([f"{item['role'].capitalize()}: {item['content']}" for item in stripped_traj])

def display_actions(actions: List[Action]) -> str:
    """
    格式化tau_bench Action序列为JSON字符串
    """
    return json.dumps([action.model_dump() for action in actions], indent=4)

def display_context(user_instruction: str, ground_truth_actions: List[Action], ground_truth_outputs: List[str], trajectory: List[Dict[str, Any]]) -> str:
    """
    构建完整的上下文信息用于LLM分析
    包含：用户指令、标准答案（动作或输出）、实际对话轨迹
    """
    traj_display = display_traj(trajectory)
    context = f"""----- start user instruction -----
{user_instruction}
----- end user instruction -----"""
    if len(ground_truth_outputs) > 0:
        context += f"""

----- start required outputs -----
{ground_truth_outputs}
----- end required outputs -----"""
    else:
        context += f"""

----- start ground truth action sequence -----
{display_actions(ground_truth_actions)}
----- end ground truth action sequence -----

----- start trajectory -----
{traj_display}
----- end trajectory -----\n"""
    return context

def fault_assignment_analysis(api: API, results: List[OriginalResult], max_concurrency: int) -> List[FaultAssignmentResult]:
    """
    错误归属分析：判断失败是由用户、智能体还是环境导致

    使用tau_bench的API对象（支持多种LLM）进行：
    1. 分类任务：将错误归属到三类之一
    2. 生成任务：为归属决策生成详细解释

    支持多线程并发处理以加速分析
    """
    def assign_fault(task_id: int, user_instruction: str, traj: List[Dict[str, Any]], ground_truth_actions: List[Action], ground_truth_outputs: List[str]) -> FaultAssignmentResult:
        """
        单个任务的错误归属分析

        使用tau_bench API的classify和generate方法
        """
        idx_to_author = {
            0: FaultAuthor.USER,
            1: FaultAuthor.AGENT,
            2: FaultAuthor.ENVIRONMENT,
        }
        # 根据tau_bench Task定义选择评估策略
        grading_strategy = GradingStrategy.OUTPUTS if len(ground_truth_outputs) > 0 else GradingStrategy.ACTIONS
        ctx_desc = context_description(grading_strategy)
        try:
            context = display_context(user_instruction, ground_truth_actions, ground_truth_outputs, traj)
        except Exception as e:
            print("task_id", task_id)
            raise(e)
        # 使用tau_bench API的classify方法进行三分类
        res = api.classify(
            instruction=f"{ctx_desc}\n\nDetermine the entity that is responsible for the fault. The user is responsible for the fault if they caused an action that was not grounded in the user instruction. The agent is responsible for the fault if they took an action that was not correct (or took the action with the wrong arguments). The environment is responsible for all other faults.",
            text=context,
            options=["The user", "The agent", "The environment (neither user nor agent)"],
        )
        author = idx_to_author[res]
        # 使用tau_bench API的generate方法生成解释
        description = api.generate(
            instruction=f"{ctx_desc}\n\nDescribe the reason why {author.value} is responsible for the fault in the trajectory. Be concise and only focus on the functional differences between the ground truth and the trajectory.",
            text=context,
        )
        return FaultAssignmentResult(task_id=task_id, author=author, description=description)
    # 多线程并发执行
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        task_ids = [r.task_id for r in results]
        user_instructions = [r.user_instruction for r in results]
        trajs = [r.traj for r in results]
        ground_truth_actions = [r.ground_truth_actions for r in results]
        ground_truth_outputs = [r.ground_truth_outputs for r in results]
        results = list(executor.map(assign_fault, task_ids, user_instructions, trajs, ground_truth_actions, ground_truth_outputs))
    return results


def fault_type_analysis(api: API, results: List[OriginalResult], max_concurrency: int) -> List[FaultTypeResult]:
    """
    错误类型分析：对智能体导致的失败进行细分

    将错误分为四类：
    1. 调用错误工具
    2. 工具参数错误
    3. 目标部分完成
    4. 其他

    使用tau_bench API进行分类和生成解释
    """
    def get_fault_type(task_id: int, user_instruction: str, traj: List[Dict[str, Any]], ground_truth_actions: List[Action], ground_truth_outputs: List[str]) -> FaultTypeResult:
        """
        单个任务的错误类型分析
        """
        idx_to_fault_type = {
            0: FaultType.CALLED_WRONG_TOOL,
            1: FaultType.USED_WRONG_TOOL_ARGUMENT,
            2: FaultType.GOAL_PARTIALLY_COMPLETED,
            3: FaultType.OTHER,
        }
        grading_strategy = GradingStrategy.OUTPUTS if len(ground_truth_outputs) > 0 else GradingStrategy.ACTIONS
        ctx_desc = context_description(grading_strategy)
        context = display_context(user_instruction, ground_truth_actions, ground_truth_outputs, traj)
        # 使用tau_bench API的classify方法进行四分类
        res = api.classify(
            instruction=f"{ctx_desc}\n\nDetermine the type of fault of the first instance of the fault.",
            text=context,
            options=["The user called the wrong tool", "The user used the correct tool with a wrong argument", "The goal was only partially completed", "Other"],
        )
        fault_type = idx_to_fault_type[res]
        # 使用tau_bench API的generate方法生成解释
        description = api.generate(
            instruction=f"{ctx_desc}\n\nDescribe the reason why the following trajectory contains a fault of type \"{fault_type.value}\". Be concise and only focus on the functional differences between the ground truth and the trajectory.",
            text=context,
        )
        return FaultTypeResult(task_id=task_id, fault_type=fault_type, description=description)
    # 多线程并发执行
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        task_ids = [r.task_id for r in results]
        user_instructions = [r.user_instruction for r in results]
        trajs = [r.traj for r in results]
        ground_truth_actions = [r.ground_truth_actions for r in results]
        ground_truth_outputs = [r.ground_truth_outputs for r in results]
        results = list(executor.map(get_fault_type, task_ids, user_instructions, trajs, ground_truth_actions, ground_truth_outputs))
    return results

def main() -> None:
    """
    主函数：自动错误识别和分析流程

    流程：
    1. 从tau_bench测试结果文件中加载数据
    2. 根据环境类型加载对应的tau_bench任务列表
    3. 筛选出失败的测试用例（reward <= 1e-3）
    4. 执行错误归属分析（用户/智能体/环境）
    5. 对智能体导致的失败进行错误类型分析
    6. 输出统计信息和详细结果
    """
    args = get_args()
    # 使用tau_bench的工具函数从命令行参数创建API实例
    api = default_api_from_args(args)
    with open(args.results_path, "r") as f:
        results = json.load(f)
    print(f"Loaded {len(results)} results")
    env = args.env
    # 根据环境类型加载tau_bench任务列表
    if env == "airline":
        tasks: List[Task] = AIRLINE_TASKS
    elif env == "retail":
        tasks: List[Task] = RETAIL_TASKS
    else:
        raise ValueError(f"Invalid environment: {env}")
    # 筛选失败的测试用例
    failed_results = [r for r in results if r["reward"] <= 1e-3]
    print(f"Found {len(failed_results)} failed trajectories")
    if args.max_num_failed_results is not None and len(failed_results) > args.max_num_failed_results:
        print(f"Limiting to {args.max_num_failed_results} failed trajectories")
        failed_results = failed_results[:args.max_num_failed_results]
    # 构建分析所需的数据结构
    original_results = []
    empty_traj = 0
    empty_task_list = []
    for result in failed_results:
        if len(result["traj"]) == 0:
            empty_traj += 1
            empty_task_list.append(result["task_id"])
            continue
        task_id: int = result["task_id"]
        # 从tau_bench任务列表获取对应的Task对象
        task = tasks[task_id]
        user_instruction = task.instruction  # tau_bench Task的指令
        ground_truth_actions = task.actions  # tau_bench Task的标准动作序列
        ground_truth_outputs = task.outputs  # tau_bench Task的期望输出
        original_result = OriginalResult(task_id=task_id, user_instruction=user_instruction, traj=result["traj"], ground_truth_actions=ground_truth_actions, ground_truth_outputs=ground_truth_outputs)
        original_results.append(original_result)
    print("Empty traj", empty_traj)
    # 执行错误归属分析
    print(f"Performing fault assignment analysis on {len(original_results)} failed trajectories with a max concurrency of {args.max_concurrency}...")
    fault_assignment_results = fault_assignment_analysis(api=api, results=original_results, max_concurrency=args.max_concurrency)
    # 筛选出智能体导致的失败
    failures_due_to_agent = [original_results[i] for i, r in enumerate(fault_assignment_results) if r.author == FaultAuthor.AGENT]
    # 对智能体导致的失败进行类型分析
    print(f"Performing fault type analysis on {len(failures_due_to_agent)} failures that have been marked as being caused by the agent with a max concurrency of {args.max_concurrency}...")
    fault_type_results = fault_type_analysis(api=api, results=failures_due_to_agent, max_concurrency=args.max_concurrency)
    # 输出统计信息
    print(f"""Reviewed {len(fault_assignment_results)} trajectories:

Author fault distribution:
  - User: {sum(1 for r in fault_assignment_results if r.author == FaultAuthor.USER)} ({round(sum(1 for r in fault_assignment_results if r.author == FaultAuthor.USER) / len(fault_assignment_results) * 100, 2)}%)
  - Agent: {sum(1 for r in fault_assignment_results if r.author == FaultAuthor.AGENT)} ({round(sum(1 for r in fault_assignment_results if r.author == FaultAuthor.AGENT) / len(fault_assignment_results) * 100, 2)}%)
  - Environment (otherwise case): {sum(1 for r in fault_assignment_results if r.author == FaultAuthor.ENVIRONMENT)} ({round(sum(1 for r in fault_assignment_results if r.author == FaultAuthor.ENVIRONMENT) / len(fault_assignment_results) * 100, 2)}%)

Fault type distribution (only failures marked as being caused by the agent):
  - Called wrong tool: {sum(1 for r in fault_type_results if r.fault_type == FaultType.CALLED_WRONG_TOOL)} ({round(sum(1 for r in fault_type_results if r.fault_type == FaultType.CALLED_WRONG_TOOL) / len(fault_type_results) * 100, 2)}%)
  - Used wrong tool argument: {sum(1 for r in fault_type_results if r.fault_type == FaultType.USED_WRONG_TOOL_ARGUMENT)} ({round(sum(1 for r in fault_type_results if r.fault_type == FaultType.USED_WRONG_TOOL_ARGUMENT) / len(fault_type_results) * 100, 2)}%)
  - Goal partially completed: {sum(1 for r in fault_type_results if r.fault_type == FaultType.GOAL_PARTIALLY_COMPLETED)} ({round(sum(1 for r in fault_type_results if r.fault_type == FaultType.GOAL_PARTIALLY_COMPLETED) / len(fault_type_results) * 100, 2)}%)
  - Other: {sum(1 for r in fault_type_results if r.fault_type == FaultType.OTHER)} ({round(sum(1 for r in fault_type_results if r.fault_type == FaultType.OTHER) / len(fault_type_results) * 100, 2)}%)
""")
    # 保存分析结果到JSON文件
    with open(args.output_path, "w") as f:
        json.dump({
            "fault_assignment_analysis": [r.model_dump() for r in fault_assignment_results],
            "fault_type_analysis": [r.model_dump() for r in fault_type_results],
            "empty_task_list": empty_task_list
        }, f, indent=4)
    print(f"Saved results to {args.output_path}")

if __name__ == "__main__":
    main()
