import json
from typing import TYPE_CHECKING
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from . import DataState


class ModifyPendingOrderAddressInput(BaseModel):
    """Input schema for modify_pending_order_address tool."""
    order_id: str = Field(
        description="The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id."
    )
    address1: str = Field(
        description="The first line of the address, such as '123 Main St'."
    )
    address2: str = Field(
        description="The second line of the address, such as 'Apt 1' or ''."
    )
    city: str = Field(
        description="The city, such as 'San Francisco'."
    )
    state: str = Field(
        description="The state, such as 'CA'."
    )
    country: str = Field(
        description="The country, such as 'USA'."
    )
    zip: str = Field(
        description="The zip code, such as '12345'."
    )


def create_modify_pending_order_address_tool(data_state: "DataState") -> StructuredTool:
    """
    Create the modify_pending_order_address tool with access to data state.

    Args:
        data_state: DataState object containing the database

    Returns:
        StructuredTool for modifying pending order addresses
    """

    def modify_pending_order_address(
        order_id: str,
        address1: str,
        address2: str,
        city: str,
        state: str,
        country: str,
        zip: str
    ) -> str:
        """
        Modify the shipping address of a pending order. The agent needs to explain the modification detail
        and ask for explicit user confirmation (yes/no) to proceed.

        Args:
            order_id: The order id
            address1: The first line of the address
            address2: The second line of the address
            city: The city
            state: The state
            country: The country
            zip: The zip code

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

        # Modify the address
        order["address"] = {
            "address1": address1,
            "address2": address2,
            "city": city,
            "state": state,
            "country": country,
            "zip": zip,
        }

        return json.dumps(order, ensure_ascii=False)

    return StructuredTool(
        name="modify_pending_order_address",
        description="Modify the shipping address of a pending order. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.",
        func=modify_pending_order_address,
        args_schema=ModifyPendingOrderAddressInput,
    )