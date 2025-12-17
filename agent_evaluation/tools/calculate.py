from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class CalculateInput(BaseModel):
    """Input schema for calculate tool."""
    expression: str = Field(
        description="The mathematical expression to calculate, such as '2 + 2'. The expression can contain numbers, operators (+, -, *, /), parentheses, and spaces."
    )


def calculate(expression: str) -> str:
    """
    Calculate the result of a mathematical expression.

    Args:
        expression: The mathematical expression to calculate

    Returns:
        The result of the calculation as a string
    """
    if not all(char in "0123456789+-*/(). " for char in expression):
        return "Error: invalid characters in expression"

    try:
        # Evaluate the mathematical expression safely
        result = str(round(float(eval(expression, {"__builtins__": None}, {})), 2))
        return result
    except Exception as e:
        return f"Error: {e}"


def create_calculate_tool() -> StructuredTool:
    """Create the calculate tool."""
    return StructuredTool(
        name="calculate",
        description="Calculate the result of a mathematical expression.",
        func=calculate,
        args_schema=CalculateInput,
    )