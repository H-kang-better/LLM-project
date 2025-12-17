import json
from typing import TYPE_CHECKING
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from . import DataState


class ModifyPendingOrderPaymentInput(BaseModel):
    """Input schema for modify_pending_order_payment tool."""
    order_id: str = Field(
        description="The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id."
    )
    payment_method_id: str = Field(
        description="The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'. These can be looked up from the user or order details."
    )


def create_modify_pending_order_payment_tool(data_state: "DataState") -> StructuredTool:
    """
    Create the modify_pending_order_payment tool with access to data state.

    Args:
        data_state: DataState object containing the database

    Returns:
        StructuredTool for modifying pending order payment methods
    """

    def modify_pending_order_payment(order_id: str, payment_method_id: str) -> str:
        """
        Modify the payment method of a pending order. The agent needs to explain the modification detail
        and ask for explicit user confirmation (yes/no) to proceed.

        Args:
            order_id: The order id
            payment_method_id: The new payment method id

        Returns:
            JSON string of order details after modification or error message
        """
        datas = data_state.get()
        orders = datas.get("orders", {})

        # Check if the order exists and is pending
        if order_id not in orders:
            return "Error: order not found"

        order = orders[order_id]
        if order["status"] != "pending":
            return "Error: non-pending order cannot be modified"

        # Check if the payment method exists
        if payment_method_id not in datas["users"][order["user_id"]]["payment_methods"]:
            return "Error: payment method not found"

        # Check that the payment history should only have one payment
        if (len(order["payment_history"]) > 1 or
            order["payment_history"][0]["transaction_type"] != "payment"):
            return "Error: there should be exactly one payment for a pending order"

        # Check that the payment method is different
        if order["payment_history"][0]["payment_method_id"] == payment_method_id:
            return "Error: the new payment method should be different from the current one"

        amount = order["payment_history"][0]["amount"]
        payment_method = datas["users"][order["user_id"]]["payment_methods"][payment_method_id]

        # Check if the new payment method has enough balance if it is a gift card
        if (payment_method["source"] == "gift_card" and
            payment_method["balance"] < amount):
            return "Error: insufficient gift card balance to pay for the order"

        # Modify the payment method
        order["payment_history"].extend([
            {
                "transaction_type": "payment",
                "amount": amount,
                "payment_method_id": payment_method_id,
            },
            {
                "transaction_type": "refund",
                "amount": amount,
                "payment_method_id": order["payment_history"][0]["payment_method_id"],
            },
        ])

        # If payment is made by gift card, update the balance
        if payment_method["source"] == "gift_card":
            payment_method["balance"] -= amount
            payment_method["balance"] = round(payment_method["balance"], 2)

        # If refund is made to a gift card, update the balance
        if "gift_card" in order["payment_history"][0]["payment_method_id"]:
            old_payment_method = datas["users"][order["user_id"]]["payment_methods"][
                order["payment_history"][0]["payment_method_id"]
            ]
            old_payment_method["balance"] += amount
            old_payment_method["balance"] = round(old_payment_method["balance"], 2)

        return json.dumps(order, ensure_ascii=False)

    return StructuredTool(
        name="modify_pending_order_payment",
        description="Modify the payment method of a pending order. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.",
        func=modify_pending_order_payment,
        args_schema=ModifyPendingOrderPaymentInput,
    )