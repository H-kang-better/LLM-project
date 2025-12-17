import json
from typing import TYPE_CHECKING, List
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from . import DataState


class ExchangeDeliveredOrderItemsInput(BaseModel):
    """Input schema for exchange_delivered_order_items tool."""
    order_id: str = Field(
        description="The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id."
    )
    item_ids: List[str] = Field(
        description="The item ids to be exchanged. IMPORTANT: These are the 10-digit item_id values from the order details (e.g., '6469567736', '8426249116'), NOT the item position numbers (1, 2, 3, etc.). You must first call get_order_details to retrieve the exact item_id values for each item you want to exchange."
    )
    new_item_ids: List[str] = Field(
        description="The NEW item ids to exchange for. IMPORTANT: These are the 10-digit variant IDs from the product catalog (e.g., '1008292230'). You must first use get_product_details or list_all_product_types to find products with the desired features, then select the correct variant ID. Each new item must be of the same product type as the corresponding old item."
    )
    payment_method_id: str = Field(
        description="The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'. These can be looked up from the user or order details."
    )


def create_exchange_delivered_order_items_tool(data_state: "DataState") -> StructuredTool:
    """
    Create the exchange_delivered_order_items tool with access to data state.

    Args:
        data_state: DataState object containing the database

    Returns:
        StructuredTool for exchanging delivered order items
    """

    def exchange_delivered_order_items(
        order_id: str,
        item_ids: List[str],
        new_item_ids: List[str],
        payment_method_id: str
    ) -> str:
        """
        Exchange items in a delivered order to new items of the same product type.
        For a delivered order, return or exchange can be only done once by the agent.
        The agent needs to explain the exchange detail and ask for explicit user confirmation (yes/no) to proceed.

        Args:
            order_id: The order id
            item_ids: The item ids to be exchanged
            new_item_ids: The new item ids
            payment_method_id: The payment method id

        Returns:
            JSON string of order details after exchange request or error message
        """
        datas = data_state.get()
        products, orders, users = datas.get("products", {}), datas.get("orders", {}), datas.get("users", {})

        # check order exists and is delivered
        if order_id not in orders:
            return "Error: order not found"

        order = orders[order_id]
        if order["status"] != "delivered":
            return "Error: non-delivered order cannot be exchanged"

        # check the items to be exchanged exist
        all_item_ids = [item["item_id"] for item in order["items"]]
        for item_id in item_ids:
            if item_ids.count(item_id) > all_item_ids.count(item_id):
                return f"Error: item_id '{item_id}' not found in order. Valid item_ids in this order are: {all_item_ids}. Please use get_order_details to retrieve the correct item_id values (10-digit numbers like '6469567736'), not item position numbers."

        # check new items exist and match old items and are available
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

        diff_price = round(diff_price, 2)

        # check payment method exists and can cover the price difference if gift card
        if payment_method_id not in users[order["user_id"]]["payment_methods"]:
            return "Error: payment method not found"

        payment_method = users[order["user_id"]]["payment_methods"][payment_method_id]
        if (payment_method["source"] == "gift_card" and
            payment_method["balance"] < diff_price):
            return "Error: insufficient gift card balance to pay for the price difference"

        # modify the order
        order["status"] = "exchange requested"
        order["exchange_items"] = sorted(item_ids)
        order["exchange_new_items"] = sorted(new_item_ids)
        order["exchange_payment_method_id"] = payment_method_id
        order["exchange_price_difference"] = diff_price

        return json.dumps(order, ensure_ascii=False)

    return StructuredTool(
        name="exchange_delivered_order_items",
        description="Exchange items in a delivered order to new items of the same product type. For a delivered order, return or exchange can be only done once by the agent. The agent needs to explain the exchange detail and ask for explicit user confirmation (yes/no) to proceed.",
        func=exchange_delivered_order_items,
        args_schema=ExchangeDeliveredOrderItemsInput,
    )