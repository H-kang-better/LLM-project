# LangChain版本的环境模块
#
# 此模块实现了兼容tau_bench框架的LangChain智能体环境
# 主要功能：
# 1. 创建LangGraph ReAct智能体（使用tau_bench工具）
# 2. 运行智能体-用户模拟器交互循环
# 3. 计算tau_bench标准的奖励（基于数据库状态和输出）
# 4. 返回tau_bench格式的评估结果

import os
import random
import copy
import uuid
from typing import List, Optional, Dict, Any

from openai import OpenAI
# tau_bench核心类型导入
from tau_bench.types import SolveResult
from tau_bench.types import (
    Action,
    Task,
    EnvInfo,
    EnvResetResponse,
    RewardResult,
    RewardOutputInfo,
    RewardActionInfo,
)
from dotenv import load_dotenv
# LangChain核心消息类型
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
# LangChain智能体构建器（新版API）
from langchain.agents import create_agent

from tools import DataState, create_all_tools
from data import load_data
from utils import generate_conversation, get_data_hash

load_dotenv(".env", override=True)


class EnvLangChain(object):
    """
    LangChain智能体的tau_bench评估环境

    此类实现了tau_bench的环境接口，但使用LangChain/LangGraph框架：
    - 使用LangGraph的create_react_agent创建智能体
    - 使用OpenAI API作为用户模拟器
    - 与tau_bench工具集成
    - 计算tau_bench标准的奖励

    与tau_bench框架兼容，返回标准的EnvRunResult
    """

    def __init__(
        self,
        tasks: List[Task],  # tau_bench的Task列表
        llm,  # LangChain LLM实例
        system_prompt: str,  # 系统提示词（包含环境知识库）
        terminate_tools: List[str] = [],  # 终止工具列表
        task_index: Optional[int] = None,  # tau_bench任务索引
        config = None  # tau_bench RunConfig
    ) -> None:
        """
        初始化LangChain环境

        Args:
            tasks: tau_bench任务列表
            llm: LangChain LLM实例
            system_prompt: 系统提示词
            terminate_tools: 触发对话结束的工具名称列表
            task_index: 要运行的tau_bench任务索引
            config: tau_bench RunConfig配置
        """
        super().__init__()
        self.llm = llm
        self.system_prompt = system_prompt
        self.terminate_tools = terminate_tools
        self.tasks = tasks

        if task_index is not None:
            self.task_index = task_index
        else:
            self.task_index = random.randint(0, len(tasks))

        self.task = tasks[self.task_index]  # 当前的tau_bench Task
        self.config = config

        # 初始化数据状态（模拟数据库）
        self.data_state = DataState(load_data())

        # 创建tau_bench工具
        self.tools = create_all_tools(self.data_state)

        # 创建LangGraph智能体
        self.agent_executor = self._create_agent()

        # 用户模拟器设置（使用OpenAI API）
        random_uuid = uuid.uuid4()
        # 根据tau_bench Task的instruction构建用户提示词
        self.user_system_prompt = self.build_user_system_prompt(self.task.instruction)
        # OpenAI客户端
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
        )
        self.user_messages = []
        self.actions: List[Action] = []  # 记录智能体执行的tau_bench Action
        self.output_list: List[str] = []  # 记录智能体输出
        self.split_message_ids = 0
        # 用户模拟器的模型
        self.model_id = config.user_model if config else "gpt-4"

        # 追踪对话
        self.chat_history: List[Dict[str, Any]] = []
        self.accumulated_usage = {
            "inputTokens": 0,
            "outputTokens": 0,
            "totalTokens": 0
        }

    def _create_agent(self):
        """
        创建LangChain ReAct智能体

        使用LangChain的create_agent构建智能体（新版API）：
        - 输入：LLM、工具列表
        - 系统提示词需要通过正确的参数传递
        - 输出：可执行的智能体

        Returns:
            LangChain智能体执行器
        """
        # 使用LangChain的create_agent（langchain.agents的新API）
        # 系统提示词通过system_prompt参数传递
        agent_executor = create_agent(
            self.llm,
            self.tools,
            system_prompt=self.system_prompt
        )
        return agent_executor

    def reset(self, task_index: Optional[int] = None) -> EnvResetResponse:
        """
        重置环境到新任务（tau_bench标准接口）

        Args:
            task_index: 新的tau_bench任务索引

        Returns:
            EnvResetResponse
        """
        if task_index is None:
            task_index = random.randint(0, len(self.tasks))
        self.task_index = task_index
        self.task = self.tasks[task_index]  # 获取新的tau_bench Task
        self.actions = []
        # 重置数据状态
        self.data_state = DataState(load_data())
        self.tools = create_all_tools(self.data_state)
        self.agent_executor = self._create_agent()
        self.chat_history = []
        self.output_list = []
        self.accumulated_usage = {
            "inputTokens": 0,
            "outputTokens": 0,
            "totalTokens": 0
        }

    def build_user_system_prompt(self, instruction: Optional[str]) -> str:
        """
        构建用户模拟器的系统提示词

        根据tau_bench Task的instruction生成提示词，指导用户模拟器：
        - 按自然方式展开对话
        - 不一次性透露所有信息
        - 不捏造未提供的信息
        - 目标达成后发送###STOP###结束对话

        Args:
            instruction: tau_bench Task的用户指令

        Returns:
            用户模拟器的系统提示词
        """
        instruction_display = (
            ("\n\nInstruction: " + instruction + "\n")
            if instruction is not None
            else ""
        )
        return f"""You are a user interacting with an agent.{instruction_display}
Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- IMPORTANT: If the instruction goal is satisfied (the agent has completed the task), you MUST immediately generate '###STOP###' as a standalone message without anything else to end the conversation. Do not continue with polite responses like "thank you" or "have a great day" after the goal is achieved.
- Do NOT enter into polite back-and-forth exchanges. Once the task is done and you say thank you, generate '###STOP###' on the next turn.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""

    def _detect_conversation_loop(self, recent_messages, threshold=3):
        """
        检测对话是否陷入重复循环

        Args:
            recent_messages: 最近的消息列表
            threshold: 判定为循环的重复次数阈值

        Returns:
            bool: 如果检测到循环返回True
        """
        if len(recent_messages) < threshold * 2:
            return False

        # 提取最近的消息内容（忽略表情符号和标点）
        def normalize(text):
            import re
            # 移除表情符号、标点和多余空格
            text = re.sub(r'[^\w\s]', '', text.lower())
            return ' '.join(text.split())

        # 检查是否有重复的对话模式
        last_messages = [normalize(msg) for msg in recent_messages[-threshold*2:]]

        # 检查最近的消息是否高度相似
        similarity_count = 0
        for i in range(len(last_messages) - 1):
            if last_messages[i] and last_messages[i] == last_messages[i + 1]:
                similarity_count += 1

        return similarity_count >= threshold - 1

    def _detect_polite_closure(self, agent_msg, user_msg):
        """
        检测对话是否进入礼貌性结束阶段

        Args:
            agent_msg: 智能体的消息
            user_msg: 用户的消息

        Returns:
            bool: 如果检测到礼貌性结束返回True
        """
        # 礼貌性结束语关键词
        closure_keywords = [
            "have a great day",
            "have a nice day",
            "you're welcome",
            "thank you",
            "thanks",
            "goodbye",
            "bye",
            "take care",
        ]

        agent_lower = agent_msg.lower()
        user_lower = user_msg.lower()

        # 检查智能体和用户的消息是否都包含结束语
        agent_has_closure = any(keyword in agent_lower for keyword in closure_keywords)
        user_has_closure = any(keyword in user_lower for keyword in closure_keywords)

        # 如果双方都使用了结束语，且消息都很短（少于50个字符），判定为礼貌性结束
        return (agent_has_closure and user_has_closure and
                len(agent_msg) < 50 and len(user_msg) < 50)

    def loop(self, max_num_steps=30):
        """
        主交互循环：智能体与用户模拟器对话

        流程：
        1. 用户模拟器发送初始消息
        2. 循环：
           a. 检查是否收到###STOP###（任务完成）
           b. 检测对话循环和礼貌性结束
           c. 智能体处理用户消息并调用工具
           d. 记录智能体的Action和输出
           e. 用户模拟器响应智能体
        3. 计算tau_bench奖励
        4. 返回SolveResult

        Args:
            max_num_steps: 最大交互步数

        Returns:
            SolveResult: tau_bench标准的结果对象
        """
        # 初始化用户消息
        self.user_messages = [{"role": "user", "content": [{"text": "Hi! How can I help you today?"}]}]
        # 用户模拟器生成第一条消息
        user_message = generate_conversation(
            self.client,
            self.model_id,
            self.user_messages,
            system_prompt=self.user_system_prompt,
            max_token=8192
        )

        # 用于检测循环的最近消息列表
        recent_messages = []
        # 连续礼貌性结束的计数
        polite_closure_count = 0

        for step in range(max_num_steps):
            reward = 0
            done = False
            # 检查用户是否发送了完成信号
            done = "###STOP###" in f"{user_message}"

            self.user_messages.append({"role": "assistant", "content": [{"text": user_message}]})

            if done:
                # 对话结束，计算奖励
                self.split_message_ids = len(self.chat_history)
                self._extract_actions_from_history()
                reward_res = self.calculate_reward()
                reward = reward_res.reward
                info = EnvInfo(task=self.task)
                info.reward_info = reward_res
                break

            # 智能体的回合
            user_input = f"{user_message}"

            try:
                # 使用LangGraph运行智能体
                # LangGraph使用messages参数而不是input/chat_history
                messages_to_agent = self.chat_history + [HumanMessage(content=user_input)]

                # 调用LangGraph智能体
                result = self.agent_executor.invoke({"messages": messages_to_agent})

                # 从LangGraph结果中提取输出
                # LangGraph在result中返回messages
                result_messages = result.get("messages", [])
                if result_messages:
                    last_message = result_messages[-1]
                    if isinstance(last_message, AIMessage):
                        agent_output = last_message.content
                    else:
                        agent_output = str(last_message)
                else:
                    agent_output = ""

                print(f"\nAgent response: {agent_output}", end="")
                # 追踪token使用（近似）
                self.accumulated_usage["inputTokens"] += len(user_input.split()) * 2
                self.accumulated_usage["outputTokens"] += len(agent_output.split()) * 2
                self.accumulated_usage["totalTokens"] = self.accumulated_usage["inputTokens"] + self.accumulated_usage["outputTokens"]

                # 从messages中提取工具调用，记录为tau_bench Action
                for msg in result_messages:
                    if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                        print(f"Function call: {msg.tool_calls}")
                        for tool_call in msg.tool_calls:
                            tool_name = tool_call.get("name", "")
                            tool_input = tool_call.get("args", {})
                            # 创建tau_bench Action
                            self.actions.append(Action(
                                name=tool_name,
                                kwargs=tool_input,
                            ))

                            # 检查终止工具
                            if tool_name in self.terminate_tools:
                                done = True

                # 更新对话历史
                self.chat_history.append(HumanMessage(content=user_input))
                self.chat_history.append(AIMessage(content=agent_output))
                self.output_list.append(agent_output)

                # 添加到最近消息列表（用于检测循环）
                recent_messages.append(agent_output)

                # 用户模拟器的回合
                self.user_messages.append({"role": "user", "content": [{"text": agent_output}]})
                user_message = generate_conversation(
                    self.client,
                    self.model_id,
                    self.user_messages,
                    system_prompt=self.user_system_prompt,
                    max_token=8192
                )

                # 添加用户消息到最近消息列表
                recent_messages.append(user_message)

                # 检测对话循环
                if self._detect_conversation_loop(recent_messages):
                    print("\n[检测到对话循环，自动结束对话]")
                    done = True

                # 检测礼貌性结束
                if self._detect_polite_closure(agent_output, user_message):
                    polite_closure_count += 1
                    print(f"\n[检测到礼貌性结束 {polite_closure_count}/2]")
                    if polite_closure_count >= 2:
                        print("\n[连续礼貌性结束，自动结束对话]")
                        done = True
                else:
                    polite_closure_count = 0  # 重置计数器

                info = EnvInfo(task=self.task)

                if done:
                    # 智能体调用了终止工具，或检测到对话循环/礼貌性结束
                    self.split_message_ids = len(self.chat_history)
                    reward_res = self.calculate_reward()
                    reward = reward_res.reward
                    info.reward_info = reward_res
                    break

            except Exception as e:
                print(f"Error in agent execution: {e}")
                import traceback
                traceback.print_exc()
                break

        # 计算成本（近似）
        total_cost = self.accumulated_usage["inputTokens"] / 1000 * 0.001 + self.accumulated_usage["outputTokens"] / 1000 * 0.005

        # 转换对话历史为tau_bench格式
        final_messages = self._convert_chat_history_to_messages()

        # 返回tau_bench标准的SolveResult
        return SolveResult(
            reward=reward,
            info=info.model_dump(),
            messages=final_messages,
            total_cost=total_cost,
        )

    def _extract_actions_from_history(self):
        """
        从对话历史中提取动作用于最终奖励计算

        注意：Action已经在loop()中实时收集，此方法保留用于兼容性
        """
        # Actions已经在循环中收集
        pass

    def _convert_chat_history_to_messages(self) -> List[Dict]:
        """
        将LangChain对话历史转换为tau_bench消息格式

        tau_bench使用的消息格式：
        {"role": "user/assistant", "content": [{"text": "..."}]}

        Returns:
            tau_bench格式的消息列表
        """
        messages = []
        for msg in self.chat_history[:self.split_message_ids]:
            if isinstance(msg, HumanMessage):
                messages.append({
                    "role": "user",
                    "content": [{"text": msg.content}]
                })
            elif isinstance(msg, AIMessage):
                messages.append({
                    "role": "assistant",
                    "content": [{"text": msg.content}]
                })

        # 添加最后的用户消息
        if self.user_messages:
            messages.append(self.user_messages[-1])

        return messages

    def calculate_reward(self) -> RewardResult:
        """
        计算tau_bench奖励

        tau_bench的奖励计算方式：
        1. 检查数据库状态是否与ground_truth匹配（通过hash比较）
        2. 检查智能体输出是否包含所有required outputs

        奖励为1.0表示完全成功，0.0表示失败

        Returns:
            RewardResult: tau_bench奖励结果对象
        """
        reward = 1.0

        # 检查数据库变更是否正确
        data_hash = get_data_hash(self.data_state.get())
        golden_data = load_data()
        actions = []

        # 在golden_data上执行tau_bench Task的标准动作序列
        for action in self.task.actions:
            if action.name not in self.terminate_tools:
                actions.append(action)
                parameters = copy.deepcopy(action.kwargs)

                # 模拟执行标准动作
                from tools import get_tool_by_name
                tool_use = {
                    "toolUseId": "123",
                    "input": parameters
                }
                tool_func = get_tool_by_name(action.name)
                if tool_func:
                    _ = tool_func(tool_use, agent=None, datas=golden_data)

        # 计算标准数据的hash
        gt_data_hash = get_data_hash(golden_data)
        # 比较智能体的数据状态和标准数据状态
        info = RewardActionInfo(
            r_actions=data_hash == gt_data_hash,
            gt_data_hash=gt_data_hash
        )

        if not info.r_actions:
            reward = 0.0

        # 检查输出（如果tau_bench Task定义了required outputs）
        if len(self.task.outputs) > 0:
            r_outputs = 1.0
            outputs = {}
            for output in self.task.outputs:
                found = False
                # 检查智能体的输出中是否包含required output
                for res in self.output_list:
                    if output.lower() in res.lower().replace(",", ""):
                        found = True
                        break
                outputs[output] = found
                if not found:
                    r_outputs = 0.0
                    reward = 0.0
            info = RewardOutputInfo(r_outputs=r_outputs, outputs=outputs)

        return RewardResult(reward=reward, info=info, actions=self.actions)