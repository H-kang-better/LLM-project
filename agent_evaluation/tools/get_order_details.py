import json
from typing import TYPE_CHECKING
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from . import DataState


class GetOrderDetailsInput(BaseModel):
    """Input schema for get_order_details tool."""
    order_id: str = Field(
        description="The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id."
    )


def create_get_order_details_tool(data_state: "DataState") -> StructuredTool:
    """
    Create the get_order_details tool with access to data state.

    Args:
        data_state: DataState object containing the database

    Returns:
        StructuredTool for getting order details
    """

    def get_order_details(order_id: str) -> str:
        """
        Get the details of an order.

        Args:
            order_id: The order id

        Returns:
            JSON string of order details or error message
        """
        datas = data_state.get()
        orders = datas.get("orders", {})

        if order_id in orders:
            return json.dumps(orders[order_id], ensure_ascii=False)

        return "Error: order not found"

    return StructuredTool(
        name="get_order_details",
        description="Get the details of an order.",
        func=get_order_details,
        args_schema=GetOrderDetailsInput,
    )