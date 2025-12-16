"""
文档处理示例：展示如何使用 doc_parse.py 和 util.py
将 HTML 文档转换为带元数据的向量数据库条目
"""

import os
from typing import List, Dict, Any
from doc_parse import enrich_chunk
from util import parse_html_intelligently, smart_chunking


def process_sec_filing(html_file_path: str) -> List[Dict[str, Any]]:
    """
    处理SEC文件的完整流程

    步骤：
    1. 解析HTML文档
    2. 智能分块
    3. 为每个块生成元数据
    4. 返回带元数据的文档块列表（可存入向量数据库）
    """
    print(f"开始处理文档: {html_file_path}")

    # 步骤1: 解析HTML
    print("  [1/3] 解析HTML文档...")
    document_sections = parse_html_intelligently(html_file_path)
    print(f"      提取了 {len(document_sections)} 个章节")

    # 步骤2: 智能分块
    print("  [2/3] 智能分块...")
    chunks = smart_chunking(document_sections, chunk_size=500)
    print(f"      生成了 {len(chunks)} 个文档块")

    # 步骤3: 为每个块生成元数据
    print("  [3/3] 生成元数据（使用LLM）...")
    enriched_chunks = []

    for i, chunk in enumerate(chunks):
        print(f"      处理块 {i+1}/{len(chunks)}...")

        # 判断是否为表格
        is_table = '<table>' in chunk['content'].lower() or '|' in chunk['content']

        # 使用 doc_parse.py 生成元数据
        metadata = enrich_chunk(chunk['content'], is_table=is_table)

        if metadata:
            enriched_chunk = {
                'content': chunk['content'],
                'source_file': html_file_path,
                'chunk_id': i,
                'metadata': metadata  # 包含 summary, keywords, hypothetical_questions
            }
            enriched_chunks.append(enriched_chunk)
        else:
            # 元数据生成失败，使用原始块
            enriched_chunks.append({
                'content': chunk['content'],
                'source_file': html_file_path,
                'chunk_id': i,
                'metadata': None
            })

    print(f"✓ 文档处理完成，共 {len(enriched_chunks)} 个增强块")
    return enriched_chunks


def save_to_vector_db(enriched_chunks: List[Dict[str, Any]]):
    """
    将增强后的文档块保存到向量数据库

    注意：这里需要实际的向量数据库连接
    可选方案：
    - Pinecone
    - Milvus
    - Qdrant
    - Chroma
    - FAISS
    """
    print("\n保存到向量数据库...")

    # 示例：使用 Chroma（需要先安装: pip install chromadb）
    # import chromadb
    # client = chromadb.Client()
    # collection = client.create_collection("financial_docs")

    for chunk in enriched_chunks:
        # 构建用于向量检索的文本
        # 组合原文 + 摘要 + 关键词，提升检索效果
        if chunk['metadata']:
            searchable_text = f"""
            {chunk['content']}

            摘要: {chunk['metadata']['summary']}
            关键词: {', '.join(chunk['metadata']['keywords'])}
            """
        else:
            searchable_text = chunk['content']

        # 存入向量数据库
        # collection.add(
        #     documents=[searchable_text],
        #     metadatas=[chunk['metadata']],
        #     ids=[f"chunk_{chunk['chunk_id']}"]
        # )

        print(f"  ✓ 块 {chunk['chunk_id']} 已保存")

    print("✓ 所有文档已保存到向量数据库")


def main():
    """
    完整示例：从HTML到向量数据库
    """
    # 示例文档路径（需要替换为实际路径）
    html_file = "example_sec_filing.html"

    if not os.path.exists(html_file):
        print(f"⚠ 示例文件 {html_file} 不存在")
        print("请提供一个实际的SEC HTML文件路径")
        return

    # 处理文档
    enriched_chunks = process_sec_filing(html_file)

    # 保存到向量数据库
    save_to_vector_db(enriched_chunks)

    # 示例：查看第一个增强块
    if enriched_chunks:
        print("\n" + "="*50)
        print("示例增强块:")
        print("="*50)
        chunk = enriched_chunks[0]
        print(f"内容预览: {chunk['content'][:200]}...")
        if chunk['metadata']:
            print(f"\n摘要: {chunk['metadata']['summary']}")
            print(f"关键词: {chunk['metadata']['keywords']}")
            print(f"可回答的问题: {chunk['metadata']['hypothetical_questions']}")


if __name__ == "__main__":
    main()