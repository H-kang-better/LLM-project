import json
from typing import TYPE_CHECKING
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from . import DataState


class GetUserDetailsInput(BaseModel):
    """Input schema for get_user_details tool."""
    user_id: str = Field(
        description="The user id, such as 'sara_doe_496'."
    )


def create_get_user_details_tool(data_state: "DataState") -> StructuredTool:
    """
    Create the get_user_details tool with access to data state.

    Args:
        data_state: DataState object containing the database

    Returns:
        StructuredTool for getting user details
    """

    def get_user_details(user_id: str) -> str:
        """
        Get the details of a user, including their orders.

        Args:
            user_id: The user id

        Returns:
            JSON string of user details or error message
        """
        datas = data_state.get()
        users = datas.get("users", {})

        if user_id in users:
            return json.dumps(users[user_id], ensure_ascii=False)

        return "Error: user not found"

    return StructuredTool(
        name="get_user_details",
        description="Get the details of a user, including their orders.",
        func=get_user_details,
        args_schema=GetUserDetailsInput,
    )