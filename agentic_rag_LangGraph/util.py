from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title


def parse_html_intelligently(file_path):
    """
    智能解析HTML，保持结构信息
    我测试过，这比普通解析效果好太多了
    """
    try:
        # 这一步很关键，infer_table_structure=True不能少
        elements = partition_html(
            filename=file_path,
            infer_table_structure=True,
            strategy='fast'  # 速度优先，准确率也够用
        )

        print(f"解析出{len(elements)}个元素，包括文本、标题、表格等")
        return [el.to_dict() for el in elements]

    except Exception as e:
        print(f"解析失败了: {e}")
        return []


# 智能分块，按标题分组，表格不会被切碎
def smart_chunking(elements):
    chunks = chunk_by_title(
        elements,
        max_characters=2048,  # 每块最大长度
        combine_text_under_n_chars=256,  # 合并小段落
        new_after_n_chars=1800  # 强制分块阈值
    )
    return chunks