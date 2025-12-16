from typing import List, Optional

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class ChunkMetadata(BaseModel):
    summary: str = Field(description="1-2句话总结这个块的内容")
    keywords: List[str] = Field(description="5-7个关键词")
    hypothetical_questions: List[str] = Field(description="这个块能回答什么问题")
    table_summary: Optional[str] = Field(description="如果是表格，用自然语言描述")


def get_enrichment_llm():
    """
    延迟初始化LLM，避免在模块导入时就需要API key
    """
    return ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(ChunkMetadata)


def enrich_chunk(chunk_content, is_table=False):
    """
    用LLM给每个文档块加上"理解层"
    这样检索的时候就不只是关键词匹配，还有语义理解
    """
    prompt = f"""
    作为财务分析专家，分析这个文档片段：
    {'这是一个表格，重点描述数据趋势' if is_table else ''}

    内容：
    {chunk_content[:3000]}  # 截断避免token超限
    """

    try:
        enrichment_llm = get_enrichment_llm()
        metadata = enrichment_llm.invoke(prompt)
        print(f"生成元数据成功: {len(metadata.keywords)}个关键词")
        return metadata.dict()
    except Exception as e:
        print(f"元数据生成失败: {e}")
        return None