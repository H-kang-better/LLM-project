import json
from typing import TYPE_CHECKING
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from . import DataState


class FindUserIdByEmailInput(BaseModel):
    """Input schema for find_user_id_by_email tool."""
    email: str = Field(
        description="The email of the user, such as 'john.doe@example.com'."
    )


def create_find_user_id_by_email_tool(data_state: "DataState") -> StructuredTool:
    """
    Create the find_user_id_by_email tool with access to data state.

    Args:
        data_state: DataState object containing the database

    Returns:
        StructuredTool for finding user ID by email
    """

    def find_user_id_by_email(email: str) -> str:
        """
        Find the user id by email. This is used to authenticate the user identity at the beginning of the conversation.

        Args:
            email: The email of the user

        Returns:
            The user id or error message
        """
        datas = data_state.get()
        users = datas.get("users", {})

        for user_id, user in users.items():
            if user.get("email", "").lower() == email.lower():
                return user_id

        return "Error: user not found"

    return StructuredTool(
        name="find_user_id_by_email",
        description="Find the user id by email. This is used to authenticate the user identity at the beginning of the conversation.",
        func=find_user_id_by_email,
        args_schema=FindUserIdByEmailInput,
    )