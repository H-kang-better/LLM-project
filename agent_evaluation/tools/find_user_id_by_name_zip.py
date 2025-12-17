import json
from typing import TYPE_CHECKING
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from . import DataState


class FindUserIdByNameZipInput(BaseModel):
    """Input schema for find_user_id_by_name_zip tool."""
    first_name: str = Field(
        description="The first name of the user, such as 'John'."
    )
    last_name: str = Field(
        description="The last name of the user, such as 'Doe'."
    )
    zip: str = Field(
        description="The zip code of the user, such as '12345'."
    )


def create_find_user_id_by_name_zip_tool(data_state: "DataState") -> StructuredTool:
    """
    Create the find_user_id_by_name_zip tool with access to data state.

    Args:
        data_state: DataState object containing the database

    Returns:
        StructuredTool for finding user ID by name and zip code
    """

    def find_user_id_by_name_zip(first_name: str, last_name: str, zip: str) -> str:
        """
        Find the user id by first name, last name, and zip code. This is used to authenticate
        the user identity at the beginning of the conversation.

        Args:
            first_name: The first name of the user
            last_name: The last name of the user
            zip: The zip code of the user

        Returns:
            The user id or error message
        """
        datas = data_state.get()
        users = datas.get("users", {})

        for user_id, user in users.items():
            if (user.get("first_name", "").lower() == first_name.lower() and
                user.get("last_name", "").lower() == last_name.lower() and
                user.get("address", {}).get("zip", "") == zip):
                return user_id

        return "Error: user not found"

    return StructuredTool(
        name="find_user_id_by_name_zip",
        description="Find the user id by first name, last name, and zip code. This is used to authenticate the user identity at the beginning of the conversation.",
        func=find_user_id_by_name_zip,
        args_schema=FindUserIdByNameZipInput,
    )