"""
节点模块
定义用于 LangGraph 工作流的各种节点
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field


class Node(BaseModel):
    """
    基础节点类
    """
    name: str = Field(description="节点名称")
    description: str = Field(description="节点描述")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行节点逻辑
        """
        raise NotImplementedError("子类必须实现 execute 方法")


class RouterNode(Node):
    """
    路由节点：根据条件决定下一步流向
    """
    routes: Dict[str, str] = Field(default_factory=dict, description="路由映射")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据状态决定路由
        """
        return {"next_node": self.routes.get("default", "end")}


class ProcessNode(Node):
    """
    处理节点：执行具体的数据处理逻辑
    """

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理数据
        """
        return {"status": "processed"}


def create_node(node_type: str, name: str, description: str, **kwargs) -> Node:
    """
    工厂函数：创建不同类型的节点
    """
    if node_type == "router":
        return RouterNode(name=name, description=description, **kwargs)
    elif node_type == "process":
        return ProcessNode(name=name, description=description)
    else:
        raise ValueError(f"未知的节点类型: {node_type}")


# 预定义的节点配置
NODE_CONFIGS = {
    "gatekeeper": {
        "type": "process",
        "description": "门卫节点：检查输入有效性"
    },
    "planner": {
        "type": "process",
        "description": "规划节点：制定执行计划"
    },
    "executor": {
        "type": "process",
        "description": "执行节点：执行具体任务"
    },
    "auditor": {
        "type": "process",
        "description": "审计节点：质量检查"
    },
    "synthesizer": {
        "type": "process",
        "description": "综合节点：生成最终结果"
    }
}


def get_node_config(node_name: str) -> Dict[str, Any]:
    """
    获取节点配置
    """
    return NODE_CONFIGS.get(node_name, {})
