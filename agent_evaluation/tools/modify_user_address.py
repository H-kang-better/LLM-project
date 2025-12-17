import json
from typing import TYPE_CHECKING
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from . import DataState


class ModifyUserAddressInput(BaseModel):
    """Input schema for modify_user_address tool."""
    user_id: str = Field(
        description="The user id, such as 'sara_doe_496'."
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


def create_modify_user_address_tool(data_state: "DataState") -> StructuredTool:
    """
    Create the modify_user_address tool with access to data state.

    Args:
        data_state: DataState object containing the database

    Returns:
        StructuredTool for modifying user addresses
    """

    def modify_user_address(
        user_id: str,
        address1: str,
        address2: str,
        city: str,
        state: str,
        country: str,
        zip: str
    ) -> str:
        """
        Modify the default address of a user. The agent needs to explain the modification detail
        and ask for explicit user confirmation (yes/no) to proceed.

        Args:
            user_id: The user id
            address1: The first line of the address
            address2: The second line of the address
            city: The city
            state: The state
            country: The country
            zip: The zip code

        Returns:
            JSON string of user details after modification or error message
        """
        datas = data_state.get()
        users = datas.get("users", {})

        if user_id not in users:
            return "Error: user not found"

        user = users[user_id]
        user["address"] = {
            "address1": address1,
            "address2": address2,
            "city": city,
            "state": state,
            "country": country,
            "zip": zip,
        }

        return json.dumps(user, ensure_ascii=False)

    return StructuredTool(
        name="modify_user_address",
        description="Modify the default address of a user. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.",
        func=modify_user_address,
        args_schema=ModifyUserAddressInput,
    )