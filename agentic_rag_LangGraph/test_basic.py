"""
简单测试脚本
测试项目的基本功能，不需要API keys
"""

import sys
import sqlite3
from db import setup_database


def test_database_setup():
    """测试数据库创建"""
    print("测试1: 数据库设置...")
    try:
        setup_database()

        # 验证数据库是否创建成功
        conn = sqlite3.connect("financials.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()

        assert len(tables) > 0, "数据库中没有表"
        print("✓ 数据库创建成功，包含表:", [t[0] for t in tables])
        return True
    except Exception as e:
        print(f"✗ 数据库测试失败: {e}")
        return False


def test_module_imports():
    """测试所有模块是否可以导入"""
    print("\n测试2: 模块导入...")
    modules = ['db', 'doc_parse', 'util', 'nodes']

    all_success = True
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}.py 导入成功")
        except Exception as e:
            print(f"✗ {module}.py 导入失败: {e}")
            all_success = False

    return all_success


def test_data_integrity():
    """测试数据库数据完整性"""
    print("\n测试3: 数据完整性...")
    try:
        conn = sqlite3.connect("financials.db")
        cursor = conn.cursor()

        # 检查数据行数
        cursor.execute("SELECT COUNT(*) FROM revenue_summary")
        count = cursor.fetchone()[0]

        # 检查数据结构
        cursor.execute("SELECT * FROM revenue_summary LIMIT 1")
        sample = cursor.fetchone()

        conn.close()

        assert count > 0, "数据表为空"
        assert len(sample) >= 4, "数据列不完整"

        print(f"✓ 数据完整性检查通过，共有 {count} 条记录")
        return True
    except Exception as e:
        print(f"✗ 数据完整性测试失败: {e}")
        return False


def test_util_functions():
    """测试工具函数"""
    print("\n测试4: 工具函数...")
    try:
        from util import parse_html_intelligently, smart_chunking
        print("✓ 工具函数导入成功")
        # 注意：实际测试需要HTML文件，这里只测试导入
        return True
    except Exception as e:
        print(f"✗ 工具函数测试失败: {e}")
        return False


def test_nodes_structure():
    """测试节点结构"""
    print("\n测试5: 节点结构...")
    try:
        from nodes import Node, RouterNode, ProcessNode, create_node, get_node_config

        # 测试创建节点
        node = create_node("process", "test_node", "测试节点")
        assert node.name == "test_node", "节点名称不正确"

        # 测试获取配置
        config = get_node_config("planner")
        assert config is not None, "无法获取节点配置"

        print("✓ 节点结构测试通过")
        return True
    except Exception as e:
        print(f"✗ 节点结构测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("="*50)
    print("开始运行项目基础测试")
    print("="*50)

    results = []
    results.append(("模块导入", test_module_imports()))
    results.append(("数据库设置", test_database_setup()))
    results.append(("数据完整性", test_data_integrity()))
    results.append(("工具函数", test_util_functions()))
    results.append(("节点结构", test_nodes_structure()))

    print("\n" + "="*50)
    print("测试结果汇总:")
    print("="*50)

    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")

    all_passed = all(r[1] for r in results)

    print("\n" + "="*50)
    if all_passed:
        print("✓ 所有基础测试通过！")
        print("\n下一步：")
        print("1. 复制 .env.example 为 .env")
        print("2. 在 .env 中配置你的 API keys")
        print("3. 运行 python main.py 开始使用")
        sys.exit(0)
    else:
        print("✗ 部分测试失败，请检查错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()