from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class TransferToHumanAgentsInput(BaseModel):
    """Input schema for transfer_to_human_agents tool."""
    summary: str = Field(
        description="A summary of the conversation to be transferred to the human agent."
    )


def transfer_to_human_agents(summary: str) -> str:
    """
    Transfer the conversation to a human agent when the request cannot be handled within the scope of your actions.

    Args:
        summary: A summary of the conversation

    Returns:
        Confirmation message
    """
    return "The conversation has been transferred to a human agent."


def create_transfer_to_human_agents_tool() -> StructuredTool:
    """Create the transfer_to_human_agents tool."""
    return StructuredTool(
        name="transfer_to_human_agents",
        description="Transfer the conversation to a human agent when the request cannot be handled within the scope of your actions.",
        func=transfer_to_human_agents,
        args_schema=TransferToHumanAgentsInput,
    )