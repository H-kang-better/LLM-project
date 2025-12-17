import json
from typing import TYPE_CHECKING, List
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from . import DataState


class ModifyPendingOrderItemsInput(BaseModel):
    """Input schema for modify_pending_order_items tool."""
    order_id: str = Field(
        description="The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id."
    )
    item_ids: List[str] = Field(
        description="The item ids to be modified, each such as '1008292230'. There could be duplicate items in the list."
    )
    new_item_ids: List[str] = Field(
        description="The item ids to be modified for, each such as '1008292230'. There could be duplicate items in the list. Each new item id should match the item id in the same position and be of the same product."
    )
    payment_method_id: str = Field(
        description="The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'. These can be looked up from the user or order details."
    )


def create_modify_pending_order_items_tool(data_state: "DataState") -> StructuredTool:
    """
    Create the modify_pending_order_items tool with access to data state.

    Args:
        data_state: DataState object containing the database

    Returns:
        StructuredTool for modifying pending order items
    """

    def modify_pending_order_items(
        order_id: str,
        item_ids: List[str],
        new_item_ids: List[str],
        payment_method_id: str
    ) -> str:
        """
        Modify items in a pending order to new items of the same product type.
        For a pending order, this function can only be called once.
        The agent needs to explain the exchange detail and ask for explicit user confirmation (yes/no) to proceed.

        Args:
            order_id: The order id
            item_ids: The item ids to be modified
            new_item_ids: The new item ids
            payment_method_id: The payment method id

        Returns:
            JSON string of order details after modification or error message
        """
        datas = data_state.get()
        products, orders, users = datas.get("products", {}), datas.get("orders", {}), datas.get("users", {})

        # Check if the order exists and is pending
        if order_id not in orders:
            return "Error: order not found"

        order = orders[order_id]
        if order["status"] != "pending":
            return "Error: non-pending order cannot be modified"

        # Check if the items to be modified exist
        all_item_ids = [item["item_id"] for item in order["items"]]
        for item_id in item_ids:
            if item_ids.count(item_id) > all_item_ids.count(item_id):
                return f"Error: {item_id} not found"

        # Check new items exist, match old items, and are available
        if len(item_ids) != len(new_item_ids):
            return "Error: the number of items to be exchanged should match"

        diff_price = 0
        for item_id, new_item_id in zip(item_ids, new_item_ids):
            item = [item for item in order["items"] if item["item_id"] == item_id][0]
            product_id = item["product_id"]
            if not (new_item_id in products[product_id]["variants"] and
                    products[product_id]["variants"][new_item_id]["available"]):
                return f"Error: new item {new_item_id} not found or available"

            old_price = item["price"]
            new_price = products[product_id]["variants"][new_item_id]["price"]
            diff_price += new_price - old_price

        # Check if the payment method exists
        if payment_method_id not in users[order["user_id"]]["payment_methods"]:
            return "Error: payment method not found"

        # If the new item is more expensive, check if the gift card has enough balance
        payment_method = users[order["user_id"]]["payment_methods"][payment_method_id]
        if (payment_method["source"] == "gift_card" and
            payment_method["balance"] < diff_price):
            return "Error: insufficient gift card balance to pay for the new item"

        # Handle the payment or refund
        order["payment_history"].append({
            "transaction_type": "payment" if diff_price > 0 else "refund",
            "amount": abs(diff_price),
            "payment_method_id": payment_method_id,
        })

        if payment_method["source"] == "gift_card":
            payment_method["balance"] -= diff_price
            payment_method["balance"] = round(payment_method["balance"], 2)

        # Modify the order
        for item_id, new_item_id in zip(item_ids, new_item_ids):
            item = [item for item in order["items"] if item["item_id"] == item_id][0]
            item["item_id"] = new_item_id
            item["price"] = products[item["product_id"]]["variants"][new_item_id]["price"]
            item["options"] = products[item["product_id"]]["variants"][new_item_id]["options"]

        order["status"] = "pending (item modified)"

        return json.dumps(order, ensure_ascii=False)

    return StructuredTool(
        name="modify_pending_order_items",
        description="Modify items in a pending order to new items of the same product type. For a pending order, this function can only be called once. The agent needs to explain the exchange detail and ask for explicit user confirmation (yes/no) to proceed.",
        func=modify_pending_order_items,
        args_schema=ModifyPendingOrderItemsInput,
    )