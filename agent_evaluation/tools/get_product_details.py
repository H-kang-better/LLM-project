import json
from typing import TYPE_CHECKING
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from . import DataState


class GetProductDetailsInput(BaseModel):
    """Input schema for get_product_details tool."""
    product_id: str = Field(
        description="The product id, such as 'P00001'."
    )


def create_get_product_details_tool(data_state: "DataState") -> StructuredTool:
    """
    Create the get_product_details tool with access to data state.

    Args:
        data_state: DataState object containing the database

    Returns:
        StructuredTool for getting product details
    """

    def get_product_details(product_id: str) -> str:
        """
        Get the details of a product.

        Args:
            product_id: The product id

        Returns:
            JSON string of product details or error message
        """
        datas = data_state.get()
        products = datas.get("products", {})

        if product_id in products:
            return json.dumps(products[product_id], ensure_ascii=False)

        return "Error: product not found"

    return StructuredTool(
        name="get_product_details",
        description="Get the details of a product.",
        func=get_product_details,
        args_schema=GetProductDetailsInput,
    )