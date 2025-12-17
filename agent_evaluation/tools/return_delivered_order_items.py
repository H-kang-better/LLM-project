import json
from typing import TYPE_CHECKING, List
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from . import DataState


class ReturnDeliveredOrderItemsInput(BaseModel):
    """Input schema for return_delivered_order_items tool."""
    order_id: str = Field(
        description="The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id."
    )
    item_ids: List[str] = Field(
        description="The item ids to be returned, each such as '1008292230'. There could be duplicate items in the list."
    )
    payment_method_id: str = Field(
        description="The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'. These can be looked up from the user or order details."
    )


def create_return_delivered_order_items_tool(data_state: "DataState") -> StructuredTool:
    """
    Create the return_delivered_order_items tool with access to data state.

    Args:
        data_state: DataState object containing the database

    Returns:
        StructuredTool for returning delivered order items
    """

    def return_delivered_order_items(
        order_id: str,
        item_ids: List[str],
        payment_method_id: str
    ) -> str:
        """
        Return some items of a delivered order. The order status will be changed to 'return requested'.
        The agent needs to explain the return detail and ask for explicit user confirmation (yes/no) to proceed.
        The user will receive follow-up email for how and where to return the item.

        Args:
            order_id: The order id
            item_ids: The item ids to be returned
            payment_method_id: The payment method id for refund

        Returns:
            JSON string of order details after return request or error message
        """
        datas = data_state.get()
        orders = datas.get("orders", {})

        # Check if the order exists and is delivered
        if order_id not in orders:
            return "Error: order not found"

        order = orders[order_id]
        if order["status"] != "delivered":
            return "Error: non-delivered order cannot be returned"

        # Check if the payment method exists and is either the original payment method or a gift card
        if payment_method_id not in datas["users"][order["user_id"]]["payment_methods"]:
            return "Error: payment method not found"

        if ("gift_card" not in payment_method_id and
            payment_method_id != order["payment_history"][0]["payment_method_id"]):
            return "Error: payment method should be either the original payment method or a gift card"

        # Check if the items to be returned exist (there could be duplicate items in either list)
        all_item_ids = [item["item_id"] for item in order["items"]]
        for item_id in item_ids:
            if item_ids.count(item_id) > all_item_ids.count(item_id):
                return "Error: some item not found"

        # Update the order status
        order["status"] = "return requested"
        order["return_items"] = sorted(item_ids)
        order["return_payment_method_id"] = payment_method_id

        return json.dumps(order, ensure_ascii=False)

    return StructuredTool(
        name="return_delivered_order_items",
        description="Return some items of a delivered order. The order status will be changed to 'return requested'. The agent needs to explain the return detail and ask for explicit user confirmation (yes/no) to proceed. The user will receive follow-up email for how and where to return the item.",
        func=return_delivered_order_items,
        args_schema=ReturnDeliveredOrderItemsInput,
    )