import time
import sqlite3
import pandas as pd
import os
from typing import TypedDict, List, Optional, Dict, Any
from dotenv import load_dotenv

from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# 加载环境变量
load_dotenv()


class AgentState(TypedDict):
    """
    智能体的状态定义
    这就是AI的"大脑内存"
    """
    original_request: str                      # 用户原始问题
    clarification_question: Optional[str]      # 澄清问题（如果需要的话）
    plan: List[str]                           # 执行计划
    intermediate_steps: List[Dict[str, Any]]  # 执行步骤记录
    verification_history: List[Dict[str, Any]] # 验证历史
    final_response: str                       # 最终回答


class QualityAudit(BaseModel):
    confidence_score: int = Field(description="置信度评分1-5")
    is_relevant: bool = Field(description="结果是否相关")
    is_consistent: bool = Field(description="数据是否一致")
    reasoning: str = Field(description="评估理由")


class DocumentLibrarian:
    def __init__(self):
        self.query_optimizer = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def optimize_query(self, user_query):
        """
        第一步：查询优化
        用户问的和数据库需要的往往不是一个语言体系
        """
        prompt = f"""
        把这个用户问题优化成更适合搜索财务文档的查询：
        原问题：{user_query}

        优化方向：使用财务术语、产品名称、风险因素等关键词
        """

        response = self.query_optimizer.invoke(prompt)
        optimized = response.content
        print(f"查询优化: '{user_query}' → '{optimized}'")
        return optimized

    def retrieve_and_rerank(self, query, top_k=5):
        """
        第二步：检索 + 重排序
        先粗筛20个候选，再精排Top5
        这样既保证召回率，又保证精确度
        """
        # 注意：这里需要实现向量数据库的连接
        # 这是一个示例框架
        print(f"检索相关文档: {query}")
        return []


class DataAnalyst:
    def __init__(self, db_path="financials.db"):
        self.db_path = db_path
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

        # 使用新的LangGraph方法，更安全可控
        toolkit = SQLDatabaseToolkit(db=self.db, llm=ChatOpenAI(model="gpt-4o"))
        tools = toolkit.get_tools()

        # 创建ReAct agent，比旧的create_sql_agent更稳定
        self.sql_agent = create_react_agent(
            model=ChatOpenAI(model="gpt-4o", temperature=0),
            tools=tools,
            state_modifier="你是一个财务数据分析专家。只执行SELECT查询，禁止修改数据。"
        )

    def query_financial_data(self, question: str) -> str:
        """
        专门处理财务数据查询
        适合具体数值问题，比如"Q4收入多少"

        注意：已经配置了只读权限，防止SQL注入
        """
        print(f"SQL分析师接到任务: {question}")

        # 添加安全检查
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        if any(keyword in question.upper() for keyword in dangerous_keywords):
            return "检测到危险操作，已拒绝执行。本系统只支持数据查询。"

        try:
            result = self.sql_agent.invoke({"messages": [("human", question)]})
            return result["messages"][-1].content
        except Exception as e:
            print(f"SQL执行出错: {e}")
            return f"抱歉，查询执行失败：{e}"

    def analyze_trends(self, question: str) -> str:
        """
        趋势分析工具
        不只是查数据，还要分析趋势
        """
        print(f"趋势分析师开始工作: {question}")

        try:
            # 先查询数据
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql("SELECT * FROM revenue_summary ORDER BY year, quarter", conn)
            conn.close()

            # 计算同比、环比
            df['period'] = df['year'].astype(str) + '-' + df['quarter']
            df['revenue_qoq'] = df['revenue_billions'].pct_change()
            df['revenue_yoy'] = df['revenue_billions'].pct_change(4)  # 4个季度=1年

            # 生成分析报告
            latest_revenue = df.iloc[-1]['revenue_billions']
            latest_qoq = df.iloc[-1]['revenue_qoq']
            latest_yoy = df.iloc[-1]['revenue_yoy']

            analysis = f"""
            最新财务趋势分析：
            - 最新季度收入：${latest_revenue}B
            - 环比增长：{latest_qoq:.1%}（vs上季度）
            - 同比增长：{latest_yoy:.1%}（vs去年同期）
            - 总体趋势：{"上升" if latest_yoy > 0 else "下降"}

            我的观察：{"增长势头还不错，但需要注意持续性" if latest_yoy > 0.1 else "增长放缓，需要关注"}
            """

            return analysis
        except Exception as e:
            return f"趋势分析失败：{e}"


class IntelligenceScout:
    def __init__(self):
        self.search_tool = TavilySearchResults(
            max_results=3,
            search_depth="advanced"  # 深度搜索模式
        )

    def search_realtime_info(self, query: str) -> str:
        """
        搜索实时信息，比如股价、新闻、竞争对手动态等
        静态文档里没有的信息就靠这个了
        """
        print(f"信息侦察开始搜索: {query}")

        try:
            search_results = self.search_tool.invoke({"query": query})

            # 格式化搜索结果
            formatted_results = []
            for result in search_results:
                formatted_results.append(f"**来源**: {result['url']}\n**内容**: {result['content'][:500]}...")

            return "\n\n".join(formatted_results)
        except Exception as e:
            return f"搜索失败：{e}"


class ReasoningEngine:
    """
    主推理引擎类
    整合所有组件
    """
    def __init__(self):
        # 初始化各种LLM
        self.ambiguity_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.planner_llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.synthesizer_llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        self.auditor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(QualityAudit)

        # 初始化工具
        self.document_librarian = DocumentLibrarian()
        self.data_analyst = DataAnalyst()
        self.intelligence_scout = IntelligenceScout()

        # 工具映射
        self.tool_map = {
            'document_librarian': self.document_librarian,
            'data_analyst': self.data_analyst,
            'intelligence_scout': self.intelligence_scout,
        }

    def ambiguity_gatekeeper(self, state: AgentState) -> Dict[str, Any]:
        """
        门卫节点：检测问题是否明确
        模糊问题直接返回澄清问题，避免瞎猜
        """
        print("门卫检查问题明确度...")

        request = state['original_request']

        # 用GPT-4o-mini快速判断
        judge_prompt = f"""
        判断这个问题是否足够明确，能够给出精确回答：

        问题："{request}"

        明确的问题举例："Q4营收多少？"、"主要竞争风险是什么？"
        模糊的问题举例："公司怎么样？"、"前景如何？"

        如果问题明确，回复"OK"
        如果模糊，给出一个澄清问题
        """

        response = self.ambiguity_llm.invoke(judge_prompt).content

        if response.strip() == "OK":
            print("问题明确，继续处理")
            return {"clarification_question": None}
        else:
            print(f"问题模糊，需要澄清：{response}")
            return {"clarification_question": response}

    def strategic_planner(self, state: AgentState) -> Dict[str, Any]:
        """
        规划师：根据问题制定执行计划
        这是整个系统的"大脑"
        """
        print("规划师开始制定执行计划...")

        request = state['original_request']

        # 工具描述，让LLM知道有什么可以用
        tools_description = """
        可用工具：
        - document_librarian: 搜索SEC文件、年报等文档
        - data_analyst: 查询具体财务数据
        - intelligence_scout: 搜索实时信息（股价、新闻等）
        """

        planning_prompt = f"""
        你是一个资深财务分析师的大脑。根据用户问题制定分析计划。

        用户问题：{request}

        {tools_description}

        制定一个Step-by-step计划，每步调用一个工具。
        最后一步必须是"FINISH"。

        输出格式：Python list
        例子：["data_analyst.query_financial_data('查询Q4收入')", "FINISH"]
        """

        plan_response = self.planner_llm.invoke(planning_prompt).content

        try:
            # 解析计划（这里用eval有风险，生产环境要用更安全的方法）
            plan = eval(plan_response)
            print(f"执行计划：{plan}")
            return {"plan": plan}
        except:
            print("计划解析失败，使用默认计划")
            return {"plan": ["FINISH"]}

    def tool_executor(self, state: AgentState) -> Dict[str, Any]:
        """
        工具执行器：按计划执行工具调用
        """
        print("⚡ 执行器开始工作...")

        plan = state['plan']
        next_step = plan[0]  # 取第一个任务

        if next_step == "FINISH":
            print("所有工具执行完毕")
            return {"plan": []}

        # 解析工具调用
        try:
            # 简单解析：tool_name.method_name('args')
            if '.' in next_step and '(' in next_step:
                parts = next_step.split('.')
                tool_name = parts[0]
                method_call = parts[1]
                method_name = method_call.split('(')[0]

                # 提取参数
                arg_start = next_step.index('(') + 1
                arg_end = next_step.rindex(')')
                tool_input = next_step[arg_start:arg_end].strip('\'"')

                print(f"  调用工具：{tool_name}.{method_name}({tool_input})")

                # 调用对应工具
                tool_obj = self.tool_map[tool_name]
                method = getattr(tool_obj, method_name)
                result = method(tool_input)

                # 记录执行结果
                step_record = {
                    'tool_name': f"{tool_name}.{method_name}",
                    'tool_input': tool_input,
                    'tool_output': result,
                    'timestamp': time.time()
                }

                current_steps = state.get('intermediate_steps', [])
                remaining_plan = plan[1:]  # 移除已执行的步骤

                return {
                    "intermediate_steps": current_steps + [step_record],
                    "plan": remaining_plan
                }
            else:
                print(f"无法解析工具调用: {next_step}")
                return {"plan": plan[1:]}

        except Exception as e:
            print(f"工具执行失败：{e}")
            return {"plan": plan[1:]}  # 跳过失败的步骤

    def quality_auditor(self, state: AgentState) -> Dict[str, Any]:
        """
        审计员：评估工具输出质量
        质量不行的话会触发重新规划
        """
        print("  审计员开始质量检查...")

        if not state.get('intermediate_steps'):
            return {"verification_history": []}

        last_step = state['intermediate_steps'][-1]
        original_request = state['original_request']

        audit_prompt = f"""
        作为质量审计员，评估工具输出质量：

        原始问题：{original_request}
        工具：{last_step['tool_name']}
        工具输出：{str(last_step['tool_output'])[:1000]}

        评估标准：
        1. 相关性：输出是否直接回答了问题？
        2. 一致性：数据是否前后一致？
        3. 完整性：信息是否充分？

        给出1-5分的置信度评分，并说明理由。
        """

        try:
            audit_result = self.auditor_llm.invoke(audit_prompt)
            print(f"  质量评分：{audit_result.confidence_score}/5")

            current_history = state.get('verification_history', [])
            return {"verification_history": current_history + [audit_result.dict()]}
        except Exception as e:
            print(f"审计失败：{e}")
            return {"verification_history": state.get('verification_history', [])}

    def conditional_router(self, state: AgentState) -> str:
        """
        条件路由器：根据当前状态决定下一步
        这是整个系统的"大脑中枢"
        """
        print("路由器分析当前状态...")

        # 1. 如果需要澄清，停止执行
        if state.get("clarification_question"):
            print("→ 路由到：等待用户澄清")
            return "__end__"

        # 2. 如果还没有计划，去制定计划
        if not state.get("plan"):
            print("→ 路由到：制定计划")
            return "planner"

        # 3. 如果质量检查失败，重新规划
        if state.get("verification_history"):
            last_audit = state["verification_history"][-1]
            if last_audit["confidence_score"] < 3:  # 评分太低
                print("→ 路由到：质量不合格，重新规划")
                # 清空计划，强制重新规划
                state['plan'] = []
                return "planner"

        # 4. 如果计划完成，进入综合分析
        if not state.get("plan") or state["plan"][0] == "FINISH":
            print("→ 路由到：综合分析")
            return "synthesizer"

        # 5. 继续执行计划
        print("→ 路由到：继续执行工具")
        return "executor"

    def strategic_synthesizer(self, state: AgentState) -> Dict[str, Any]:
        """
        策略师：综合所有信息，生成有洞察力的回答
        不只是总结，还要提出假设和连接
        """
        print("策略师开始综合分析...")

        request = state['original_request']
        all_evidence = state['intermediate_steps']

        # 构建上下文
        context_parts = []
        for step in all_evidence:
            context_parts.append(f"**{step['tool_name']}的发现**：\n{step['tool_output']}\n")

        full_context = "\n".join(context_parts)

        synthesis_prompt = f"""
        作为资深分析师，基于收集的信息给出深度分析：

        用户问题：{request}

        收集的证据：
        {full_context}

        要求：
        1. 首先直接回答用户问题
        2. 然后进行深度分析：寻找不同信息之间的关联
        3. 提出数据支撑的假设或洞察
        4. 保持分析的客观性，但要有个人观点

        记住：你不是在总结信息，而是在进行分析和推理。
        """

        final_answer = self.synthesizer_llm.invoke(synthesis_prompt).content
        print("综合分析完成")

        return {"final_response": final_answer}

    def build_graph(self):
        """
        构建完整的推理引擎图
        """
        # 创建状态图
        graph = StateGraph(AgentState)

        # 添加所有节点
        graph.add_node("gatekeeper", self.ambiguity_gatekeeper)
        graph.add_node("planner", self.strategic_planner)
        graph.add_node("executor", self.tool_executor)
        graph.add_node("auditor", self.quality_auditor)
        graph.add_node("synthesizer", self.strategic_synthesizer)

        # 设置入口点
        graph.set_entry_point("gatekeeper")

        # 定义路由逻辑
        graph.add_conditional_edges(
            "gatekeeper",
            lambda state: "planner" if state.get("clarification_question") is None else END
        )

        graph.add_edge("planner", "executor")
        graph.add_edge("executor", "auditor")

        # 核心路由逻辑
        graph.add_conditional_edges("auditor", self.conditional_router, {
            "planner": "planner",
            "executor": "executor",
            "synthesizer": "synthesizer",
            "__end__": END
        })

        graph.add_edge("synthesizer", END)

        # 编译图
        app = graph.compile()
        print("推理引擎构建完成！")

        return app


def main():
    """
    主函数：演示如何使用这个系统
    """
    from db import setup_database

    # 1. 初始化数据库
    print("=" * 50)
    print("初始化数据库...")
    setup_database()

    # 2. 创建推理引擎
    print("\n" + "=" * 50)
    print("构建推理引擎...")
    engine = ReasoningEngine()
    app = engine.build_graph()

    # 3. 运行查询示例
    print("\n" + "=" * 50)
    print("开始处理查询...")

    initial_state = {
        "original_request": "2023年Q4的收入是多少？",
        "clarification_question": None,
        "plan": [],
        "intermediate_steps": [],
        "verification_history": [],
        "final_response": ""
    }

    try:
        result = app.invoke(initial_state)

        print("\n" + "=" * 50)
        print("最终回答：")
        print(result.get("final_response", "没有生成回答"))

        if result.get("clarification_question"):
            print("\n需要澄清的问题：")
            print(result["clarification_question"])

    except Exception as e:
        print(f"\n执行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
