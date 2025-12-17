import json
from typing import TYPE_CHECKING
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

if TYPE_CHECKING:
    from . import DataState


class ListAllProductTypesInput(BaseModel):
    """Input schema for list_all_product_types tool (no parameters needed)."""
    pass


def create_list_all_product_types_tool(data_state: "DataState") -> StructuredTool:
    """
    Create the list_all_product_types tool with access to data state.

    Args:
        data_state: DataState object containing the database

    Returns:
        StructuredTool for listing all product types
    """

    def list_all_product_types() -> str:
        """
        List all product types.

        Returns:
            JSON string of all product types
        """
        datas = data_state.get()
        products = datas.get("products", {})

        product_types = {}
        for product_id, product in products.items():
            product_type = product.get("product_type", "")
            if product_type not in product_types:
                product_types[product_type] = []
            product_types[product_type].append(product_id)

        return json.dumps(product_types, ensure_ascii=False)

    return StructuredTool(
        name="list_all_product_types",
        description="List all product types.",
        func=list_all_product_types,
        args_schema=ListAllProductTypesInput,
    )