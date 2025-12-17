from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class ThinkInput(BaseModel):
    """Input schema for think tool."""
    thought: str = Field(
        description="A thought to think about."
    )


def think(thought: str) -> str:
    """
    Use the tool to think about something. It will not obtain new information or change the database,
    but just append the thought to the log. Use it when complex reasoning or some cache memory is needed.

    Args:
        thought: A thought to think about

    Returns:
        Empty string
    """
    # This method does not change the state of the data; it simply returns an empty string.
    return ""


def create_think_tool() -> StructuredTool:
    """Create the think tool."""
    return StructuredTool(
        name="think",
        description="Use the tool to think about something. It will not obtain new information or change the database, but just append the thought to the log. Use it when complex reasoning or some cache memory is needed.",
        func=think,
        args_schema=ThinkInput,
    )