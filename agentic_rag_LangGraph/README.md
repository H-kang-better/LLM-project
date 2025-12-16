# Agentic RAG Demo

这是一个基于 LangGraph 的智能 RAG（Retrieval-Augmented Generation）系统，用于财务分析和问答。

## 功能特点

- **智能问题检测**: 自动检测问题是否明确，模糊问题会要求澄清
- **动态执行计划**: 根据问题自动制定执行计划
- **多工具协同**: 整合文档检索、SQL查询、趋势分析和网络搜索
- **质量审计**: 自动评估输出质量，确保结果可靠
- **深度分析**: 不仅提供答案，还进行关联分析和洞察

## 系统架构

```
用户问题 → 门卫检查 → 规划制定 → 工具执行 → 质量审计 → 综合分析 → 最终回答
                ↑____________↓ (质量不合格则重新规划)
```

### 核心组件

1. **DocumentLibrarian** - 文档检索和重排序
2. **DataAnalyst** - SQL数据查询和趋势分析
3. **IntelligenceScout** - 实时信息搜索
4. **ReasoningEngine** - 主推理引擎，协调所有组件

## 安装步骤

### 1. 克隆项目

```bash
cd agentic-rag-demo
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制 `.env.example` 为 `.env` 并填入你的API密钥：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

获取API密钥：
- OpenAI API: https://platform.openai.com/api-keys
- Tavily API: https://tavily.com/

## 运行项目

### 基础运行

```bash
python main.py
```

### 自定义查询

编辑 `main.py` 中的查询示例：

```python
initial_state = {
    "original_request": "你的问题",  # 修改这里
    "clarification_question": None,
    "plan": [],
    "intermediate_steps": [],
    "verification_history": [],
    "final_response": ""
}
```

## 项目结构

```
agentic-rag-demo/
├── main.py              # 主程序入口
├── db.py                # 数据库设置
├── doc_parse.py         # 文档解析和元数据生成
├── util.py              # 工具函数
├── nodes.py             # 节点定义
├── requirements.txt     # Python依赖
├── .env.example         # 环境变量模板
├── .env                 # 环境变量配置（需要创建）
└── financials.db        # SQLite数据库（自动生成）
```

## 使用示例

### 示例1: 查询财务数据

```python
"2023年Q4的收入是多少？"
```

输出：
- SQL查询结果
- 趋势分析
- 综合分析报告

### 示例2: 趋势分析

```python
"最近的收入趋势如何？"
```

输出：
- 环比和同比增长率
- 趋势判断
- 观察和建议

### 示例3: 实时信息

```python
"最新的股价和新闻是什么？"
```

输出：
- 网络搜索结果
- 综合信息分析

## 数据库说明

项目使用SQLite存储模拟的财务数据。数据库在首次运行时自动创建，包含：

- **revenue_summary** 表：季度收入和净利润数据

你可以通过修改 `db.py` 来添加更多数据。

## 常见问题

### Q1: 提示"API key not found"

确保已创建 `.env` 文件并正确配置了API密钥。

### Q2: 导入错误

确保已安装所有依赖：
```bash
pip install -r requirements.txt
```

### Q3: 模型调用失败

检查：
1. API密钥是否正确
2. 网络连接是否正常
3. API额度是否充足

### Q4: 想使用其他LLM模型

在 `ReasoningEngine.__init__()` 中修改模型名称：
```python
self.planner_llm = ChatOpenAI(model="gpt-4o", temperature=0)  # 改成其他模型
```

## 扩展功能

### 添加新工具

1. 在相应类中定义新方法
2. 在 `ReasoningEngine.tool_map` 中注册
3. 更新规划提示词，让LLM知道新工具

### 添加新数据源

1. 修改 `db.py` 添加新表
2. 在 `DataAnalyst` 中添加查询方法

### 自定义节点逻辑

修改 `ReasoningEngine` 中的各个节点方法来自定义行为。

## 技术栈

- **LangChain**: LLM应用框架
- **LangGraph**: 工作流编排
- **OpenAI GPT-4**: 主LLM
- **Sentence Transformers**: 向量嵌入和重排序
- **Tavily**: 网络搜索
- **SQLite**: 数据存储
- **Pandas**: 数据分析

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题，请提交 Issue。
