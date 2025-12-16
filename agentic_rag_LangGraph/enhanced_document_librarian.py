"""
升级版 DocumentLibrarian：集成文档增强和向量检索

这个文件展示了如何将 doc_parse.py 集成到 DocumentLibrarian 类中
"""

import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from doc_parse import enrich_chunk

# 示例：使用 Chroma 向量数据库（可替换为其他向量数据库）
try:
    import chromadb
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False
    print("⚠ chromadb 未安装，向量检索功能不可用")
    print("安装方法: pip install chromadb")


class EnhancedDocumentLibrarian:
    """
    升级版文档管理员：支持文档增强和向量检索
    """

    def __init__(self, llm):
        self.query_optimizer = llm

        # 初始化向量数据库
        if VECTOR_DB_AVAILABLE:
            self.chroma_client = chromadb.Client()
            try:
                self.collection = self.chroma_client.get_collection("financial_docs")
                print("✓ 连接到现有向量数据库")
            except:
                self.collection = self.chroma_client.create_collection("financial_docs")
                print("✓ 创建新的向量数据库")
        else:
            self.collection = None

    def index_document(self, file_path: str):
        """
        索引新文档到向量数据库

        使用场景：
        - 新增 SEC 文件时调用
        - 定期批量索引文档
        """
        if not self.collection:
            print("⚠ 向量数据库不可用，无法索引文档")
            return

        print(f"开始索引文档: {file_path}")

        # 这里应该使用 util.py 的解析和分块功能
        # 为简化示例，这里直接读取文件
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 简单分块（实际应该使用 smart_chunking）
            chunk_size = 1000
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

            for i, chunk_content in enumerate(chunks):
                # 使用 doc_parse.py 生成元数据
                metadata = enrich_chunk(chunk_content)

                if metadata:
                    # 构建可搜索文本（原文 + 元数据）
                    searchable_text = f"""
                    {chunk_content}

                    摘要: {metadata['summary']}
                    关键词: {', '.join(metadata['keywords'])}
                    """

                    # 存入向量数据库
                    self.collection.add(
                        documents=[searchable_text],
                        metadatas=[{
                            'source': file_path,
                            'chunk_id': i,
                            'summary': metadata['summary'],
                            'keywords': ','.join(metadata['keywords'])
                        }],
                        ids=[f"{os.path.basename(file_path)}_chunk_{i}"]
                    )

                    print(f"  ✓ 索引块 {i+1}/{len(chunks)}")

            print(f"✓ 文档索引完成: {len(chunks)} 个块")

        except Exception as e:
            print(f"✗ 文档索引失败: {e}")

    def optimize_query(self, user_query: str) -> str:
        """
        查询优化：将用户问题转换为更适合搜索的查询
        """
        prompt = f"""
        把这个用户问题优化成更适合搜索财务文档的查询：
        原问题：{user_query}

        优化方向：使用财务术语、产品名称、风险因素等关键词
        只返回优化后的查询，不要其他内容。
        """

        response = self.query_optimizer.invoke(prompt)
        optimized = response.content.strip()
        print(f"查询优化: '{user_query}' → '{optimized}'")
        return optimized

    def retrieve_and_rerank(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        检索 + 重排序
        """
        if not self.collection:
            print("⚠ 向量数据库不可用")
            return []

        # 优化查询
        optimized_query = self.optimize_query(query)

        # 向量检索
        print(f"检索相关文档: {optimized_query}")
        try:
            results = self.collection.query(
                query_texts=[optimized_query],
                n_results=top_k
            )

            # 格式化结果
            documents = []
            if results['documents']:
                for i, (doc, metadata) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0]
                )):
                    documents.append({
                        'rank': i + 1,
                        'content': doc,
                        'source': metadata.get('source', 'unknown'),
                        'summary': metadata.get('summary', ''),
                        'keywords': metadata.get('keywords', '').split(',')
                    })

            print(f"✓ 检索到 {len(documents)} 个相关文档")
            return documents

        except Exception as e:
            print(f"✗ 检索失败: {e}")
            return []

    def search_sec_filings(self, query: str) -> str:
        """
        搜索SEC文件和财务报告
        这是供 ReasoningEngine 调用的工具方法
        """
        print(f"SEC文档搜索: {query}")

        # 检索相关文档
        documents = self.retrieve_and_rerank(query, top_k=3)

        if not documents:
            return "暂未检索到相关SEC文档。建议使用 data_analyst 工具查询财务数据。"

        # 格式化返回结果
        results = []
        for doc in documents:
            results.append(f"""
【来源】: {doc['source']}
【摘要】: {doc['summary']}
【内容片段】: {doc['content'][:300]}...
            """.strip())

        return "\n\n" + "\n\n".join(results)


def demo_usage():
    """
    演示如何使用增强版 DocumentLibrarian
    """
    from dotenv import load_dotenv
    load_dotenv()

    # 初始化 LLM
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "qwen-turbo"),
        temperature=0,
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY")
    )

    # 创建文档管理员
    librarian = EnhancedDocumentLibrarian(llm)

    # 步骤1: 索引文档（首次使用时需要）
    # librarian.index_document("path/to/sec_filing.html")

    # 步骤2: 搜索文档
    result = librarian.search_sec_filings("2023年Q4的收入情况")
    print("\n" + "="*50)
    print("搜索结果:")
    print("="*50)
    print(result)


if __name__ == "__main__":
    demo_usage()