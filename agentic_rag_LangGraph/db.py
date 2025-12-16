import pandas as pd
import sqlite3

# 模拟真实的财务数据
financial_data = {
    'year': [2023]*4 + [2022]*4,
    'quarter': ['Q4', 'Q3', 'Q2', 'Q1'] * 2,
    'revenue_billions': [61.9, 56.5, 52.9, 52.7, 51.9, 50.1, 49.4, 51.7],
    'net_income_billions': [21.9, 22.3, 17.4, 16.4, 17.6, 16.7, 16.7, 18.8]
}

df = pd.DataFrame(financial_data)

# 存储到SQLite，方便SQL agent查询
def setup_database():
    conn = sqlite3.connect("financials.db")
    df.to_sql("revenue_summary", conn, if_exists="replace", index=False)
    conn.close()
    print("数据库搞定了，SQL agent可以直接查询")