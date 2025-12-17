# Agent Evaluation - LangChain 版本

基于 LangChain 框架的智能体评估系统，专门用于评估零售客服智能体在处理客户服务任务时的性能表现。

## 概述

本项目是原 Strands 框架实现的 LangChain 版本改写，保持与 tau-bench 框架的完全兼容性，同时利用 LangChain 生态系统的强大功能。

## 主要变更

### 架构变更

1. **Agent 实现**
   - 从 `strands.Agent` 迁移到 `langchain.agents.AgentExecutor`
   - 使用 `create_tool_calling_agent` 创建工具调用智能体
   - 支持 LangChain 的消息历史和状态管理

2. **工具系统**
   - 从 Strands 工具格式转换为 LangChain `StructuredTool`
   - 使用 Pydantic 模型定义工具输入 schema
   - 通过 `DataState` 类管理共享状态

3. **模型支持**
   - `BedrockModel` → `ChatBedrock`
   - `OpenAIModel` → `ChatOpenAI`
   - `LiteLLMModel` → `ChatLiteLLM`
   - 支持所有 LangChain 兼容的模型提供商

4. **追踪和监控**
   - 支持 Langfuse 回调处理器
   - 兼容 LangSmith 追踪
   - 保持原有的遥测数据收集

## 项目结构

```
.
├── tools_langchain/          # LangChain 版本的工具模块
│   ├── __init__.py           # 工具工厂和 DataState 类
│   ├── calculate.py          # 计算工具
│   ├── get_user_details.py   # 获取用户详情工具
│   ├── cancel_pending_order.py  # 取消订单工具
│   └── ...                   # 其他 16 个工具
├── env_langchain.py          # LangChain 版本的环境模块
├── run_langchain.py          # LangChain 版本的运行模块
├── main_langchain.py         # LangChain 版本的主入口
├── requirements_langchain.txt # LangChain 依赖
└── README_LANGCHAIN.md       # 本文档
```

## 快速开始

### 1. 环境要求

```bash
# 安装 tau-bench
git clone https://github.com/sierra-research/tau-bench && cd ./tau-bench
pip install -e .
cd ../

# 返回项目目录
cd agent-evaluation

# 安装 LangChain 版本依赖
pip install -r requirements_langchain.txt
```

### 2. 环境配置

创建 `.env` 文件并配置以下变量：

```bash
# AWS Bedrock (如果使用 Bedrock 模型)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION_NAME=us-west-2

# OpenAI 或兼容 API
API_KEY=your_api_key
API_URL=https://api.openai.com/v1

# Langfuse (可选)
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com

# DashScope (如果使用阿里云模型)
DASHSCOPE_API_KEY=your_dashscope_key
```

### 3. 运行评估

#### 使用 OpenAI 模型

```bash
python main_langchain.py \
  --agent-strategy tool-calling \
  --env retail \
  --model gpt-4 \
  --model-provider openai \
  --user-model gpt-4o \
  --user-model-provider openai \
  --temperature 0.0 \
  --max-concurrency 1 \
  --start-index 0 \
  --end-index 5
```

#### 使用 AWS Bedrock Claude

```bash
python main_langchain.py \
  --agent-strategy tool-calling \
  --env retail \
  --model anthropic.claude-3-sonnet-20240229-v1:0 \
  --model-provider bedrock \
  --user-model anthropic.claude-3-haiku-20240307-v1:0 \
  --user-model-provider bedrock \
  --temperature 0.0 \
  --max-concurrency 1
```

#### 使用 LiteLLM (支持多个提供商)

```bash
python main_langchain.py \
  --agent-strategy tool-calling \
  --env retail \
  --model dashscope/qwen-max \
  --model-provider dashscope \
  --user-model dashscope/qwen-plus \
  --user-model-provider dashscope \
  --temperature 0.0
```

### 4. 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 智能体使用的模型 | 必填 |
| `--model-provider` | 模型提供商 (openai, bedrock, litellm 等) | openai |
| `--user-model` | 用户模拟器使用的模型 | gpt-4o |
| `--user-model-provider` | 用户模拟器模型提供商 | openai |
| `--env` | 评估环境 (retail, airline) | retail |
| `--task-split` | 任务集 (train, test, dev) | test |
| `--temperature` | 采样温度 | 0.0 |
| `--start-index` | 起始任务索引 | 0 |
| `--end-index` | 结束任务索引 (-1 表示全部) | -1 |
| `--task-ids` | 指定运行的任务 ID 列表 | None |
| `--num-trials` | 每个任务的试验次数 | 1 |
| `--log-dir` | 结果保存目录 | results |
| `--seed` | 随机种子 | 10 |

## 核心组件说明

### 1. DataState 状态管理

`DataState` 类用于在工具之间共享数据库状态：

```python
from tools_langchain import DataState, create_all_tools

# 创建状态
data_state = DataState(initial_data)

# 创建工具
tools = create_all_tools(data_state)

# 工具可以读写共享状态
data = data_state.get()
data_state.set(modified_data)
```

### 2. 工具定义

每个工具都是一个独立的模块，使用 LangChain 的 `StructuredTool`：

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MyToolInput(BaseModel):
    param: str = Field(description="参数描述")

def my_tool_func(param: str) -> str:
    # 工具实现
    return result

def create_my_tool(data_state: DataState) -> StructuredTool:
    def wrapped_func(param: str) -> str:
        # 可以访问 data_state
        data = data_state.get()
        # 处理逻辑
        return result

    return StructuredTool(
        name="my_tool",
        description="工具描述",
        func=wrapped_func,
        args_schema=MyToolInput,
    )
```

### 3. 环境类 (EnvLangChain)

环境类负责：
- 创建和管理 LangChain Agent
- 处理用户模拟器对话
- 计算任务奖励
- 保持与 tau-bench 的兼容性

```python
env = EnvLangChain(
    tasks=tasks,
    llm=llm,
    system_prompt=system_prompt,
    terminate_tools=["transfer_to_human_agents"],
    task_index=idx,
    config=config
)
result = env.loop()
```

## 与原版本的兼容性

LangChain 版本完全兼容 tau-bench 框架：

- ✅ 保持相同的任务定义格式
- ✅ 使用相同的评估指标 (Reward, Pass@k)
- ✅ 输出格式与原版本一致
- ✅ 支持相同的工具集
- ✅ 保持相同的系统提示词

## 性能对比

| 指标 | Strands 版本 | LangChain 版本 |
|------|-------------|---------------|
| 工具数量 | 17 | 17 |
| 模型支持 | Bedrock, OpenAI, LiteLLM | 所有 LangChain 支持的模型 |
| 追踪 | Langfuse (OTEL) | Langfuse, LangSmith |
| 状态管理 | agent.state | DataState 类 |
| 可扩展性 | 中等 | 高 (LangChain 生态) |

## 调试和开发

### 启用详细日志

在环境变量中设置：

```bash
export LANGCHAIN_VERBOSE=true
export LANGCHAIN_TRACING_V2=true
```

### 查看工具调用

在 `env_langchain.py` 中设置：

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=self.tools,
    verbose=True,  # 启用详细输出
    ...
)
```

## 故障排除

### 1. 模型连接错误

确保正确配置了 API 密钥和端点：

```bash
# 检查环境变量
echo $API_KEY
echo $API_URL
```

### 2. 工具执行错误

检查 `DataState` 是否正确传递：

```python
# 在工具函数中添加调试
def my_tool(param: str) -> str:
    data = data_state.get()
    print(f"Current data keys: {data.keys()}")
    ...
```

### 3. 奖励计算不匹配

验证数据库状态哈希：

```python
from utils import get_data_hash

current_hash = get_data_hash(data_state.get())
expected_hash = get_data_hash(golden_data)
print(f"Current: {current_hash}")
print(f"Expected: {expected_hash}")
```

## 扩展和定制

### 添加新工具

1. 在 `tools_langchain/` 创建新工具文件
2. 定义 Pydantic 输入模型
3. 实现工具函数
4. 在 `__init__.py` 中注册

### 使用自定义模型

```python
from langchain_community.chat_models import ChatCustomModel

llm = ChatCustomModel(
    model_name="your-model",
    temperature=0.0,
    # 其他配置
)
```

### 自定义评估指标

修改 `env_langchain.py` 中的 `calculate_reward` 方法。

## 参考资料

- [LangChain 文档](https://python.langchain.com/)
- [tau-bench 框架](https://github.com/sierra-research/tau-bench)
- [原 Strands 版本 README](./README.md)

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

与原项目保持一致。