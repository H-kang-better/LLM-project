# LangChain version of tools

from typing import Dict, Any, List
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# Import all tool creation functions
from .calculate import create_calculate_tool, calculate
from .cancel_pending_order import create_cancel_pending_order_tool
from .exchange_delivered_order_items import create_exchange_delivered_order_items_tool
from .find_user_id_by_email import create_find_user_id_by_email_tool
from .find_user_id_by_name_zip import create_find_user_id_by_name_zip_tool
from .get_order_details import create_get_order_details_tool
from .get_product_details import create_get_product_details_tool
from .get_user_details import create_get_user_details_tool
from .list_all_product_types import create_list_all_product_types_tool
from .modify_pending_order_address import create_modify_pending_order_address_tool
from .modify_pending_order_items import create_modify_pending_order_items_tool
from .modify_pending_order_payment import create_modify_pending_order_payment_tool
from .modify_user_address import create_modify_user_address_tool
from .return_delivered_order_items import create_return_delivered_order_items_tool
from .think import create_think_tool, think
from .transfer_to_human_agents import create_transfer_to_human_agents_tool, transfer_to_human_agents


class DataState:
    """
    Wrapper class to hold mutable state (database) that can be shared across tools.
    This allows tools to read and modify the database state.
    """
    def __init__(self, datas: Dict[str, Any]):
        self.datas = datas

    def get(self) -> Dict[str, Any]:
        return self.datas

    def set(self, datas: Dict[str, Any]):
        self.datas = datas


def create_all_tools(data_state: DataState) -> List[StructuredTool]:
    """
    Create all tools with the given data state.

    Args:
        data_state: DataState object containing the mutable database

    Returns:
        List of LangChain StructuredTool objects
    """
    tools = [
        create_calculate_tool(),
        create_cancel_pending_order_tool(data_state),
        create_exchange_delivered_order_items_tool(data_state),
        create_find_user_id_by_email_tool(data_state),
        create_find_user_id_by_name_zip_tool(data_state),
        create_get_order_details_tool(data_state),
        create_get_product_details_tool(data_state),
        create_get_user_details_tool(data_state),
        create_list_all_product_types_tool(data_state),
        create_modify_pending_order_address_tool(data_state),
        create_modify_pending_order_items_tool(data_state),
        create_modify_pending_order_payment_tool(data_state),
        create_modify_user_address_tool(data_state),
        create_return_delivered_order_items_tool(data_state),
        create_think_tool(),
        create_transfer_to_human_agents_tool(),
    ]
    return tools


__all__ = [
    'DataState',
    'create_all_tools',
    'create_calculate_tool',
    'create_cancel_pending_order_tool',
    'create_exchange_delivered_order_items_tool',
    'create_find_user_id_by_email_tool',
    'create_find_user_id_by_name_zip_tool',
    'create_get_order_details_tool',
    'create_get_product_details_tool',
    'create_get_user_details_tool',
    'create_list_all_product_types_tool',
    'create_modify_pending_order_address_tool',
    'create_modify_pending_order_items_tool',
    'create_modify_pending_order_payment_tool',
    'create_modify_user_address_tool',
    'create_return_delivered_order_items_tool',
    'create_think_tool',
    'create_transfer_to_human_agents_tool',
]

def get_tool_by_name(tool_name: str):
    """
    根据工具名称获取对应的工具函数（用于非LangChain模式的直接调用）

    此函数用于env.py中的奖励计算，需要在golden_data上模拟执行ground truth动作。
    返回的函数接受旧的工具调用接口：(tool_use, agent, datas)

    Args:
        tool_name: 工具名称

    Returns:
        适配器函数，接受(tool_use, agent, datas)参数
    """
    # 无状态工具：直接使用导入的函数
    stateless_tools = {
        "calculate": calculate,
        "think": think,
        "transfer_to_human_agents": transfer_to_human_agents,
    }

    if tool_name in stateless_tools:
        func = stateless_tools[tool_name]

        def wrapper(tool_use, agent=None, datas=None):
            """无状态工具的适配器"""
            params = tool_use.get("input", {})
            return func(**params)

        return wrapper

    # 有状态工具：为每次调用创建临时DataState
    stateful_tool_creators = {
        "cancel_pending_order": create_cancel_pending_order_tool,
        "exchange_delivered_order_items": create_exchange_delivered_order_items_tool,
        "find_user_id_by_email": create_find_user_id_by_email_tool,
        "find_user_id_by_name_zip": create_find_user_id_by_name_zip_tool,
        "get_order_details": create_get_order_details_tool,
        "get_product_details": create_get_product_details_tool,
        "get_user_details": create_get_user_details_tool,
        "list_all_product_types": create_list_all_product_types_tool,
        "modify_pending_order_address": create_modify_pending_order_address_tool,
        "modify_pending_order_items": create_modify_pending_order_items_tool,
        "modify_pending_order_payment": create_modify_pending_order_payment_tool,
        "modify_user_address": create_modify_user_address_tool,
        "return_delivered_order_items": create_return_delivered_order_items_tool,
    }

    if tool_name in stateful_tool_creators:
        creator = stateful_tool_creators[tool_name]

        def wrapper(tool_use, agent=None, datas=None):
            """
            有状态工具的适配器

            为每次调用创建临时DataState，这样工具可以修改datas字典
            """
            if datas is None:
                raise ValueError(f"Tool '{tool_name}' requires datas parameter")

            # 创建临时DataState包装器
            temp_data_state = DataState(datas)

            # 使用临时DataState创建工具
            tool = creator(temp_data_state)

            # 提取参数并调用工具函数
            params = tool_use.get("input", {})
            return tool.func(**params)

        return wrapper

    # 工具未找到
    print(f"Warning: Tool '{tool_name}' not found")
    return None