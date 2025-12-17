# data 文件夹内容生成分析

## 概述

`data` 文件夹包含了用于 tau_bench 零售领域（retail）智能体评估的模拟数据库。这些数据是为了支持客服智能体（customer service agent）的测试任务而生成的。

## 文件结构

```
data/
├── __init__.py           # 数据加载模块
├── users.json            # 用户数据库
├── products.json         # 产品数据库
├── orders.json           # 订单数据库
├── readme.md             # 英文说明
└── readme_zh.md          # 中文说明
```

## 与 tau_bench 的关系

### tau_bench.envs.retail.tasks_test.TASKS_TEST

这个导入来自 tau_bench 官方库，提供了一系列预定义的**零售客服测试任务**：

```python
from tau_bench.envs.retail.tasks_test import TASKS_TEST

# TASKS_TEST 包含多个测试任务，每个任务有：
# - instruction: 用户指令（如"我要取消订单"）
# - actions: 期望智能体执行的标准动作序列
# - outputs: 期望智能体输出的关键信息
```

### 数据与任务的匹配

`data` 文件夹中的数据是**专门为这些测试任务设计**的：

1. **任务需要什么数据，就生成什么数据**
2. **数据的结构必须支持所有测试任务的执行**
3. **数据量要足够让任务有足够的测试覆盖**

## 数据生成方法论

根据 `readme_zh.md` 的说明，数据生成遵循三个阶段：

### 阶段1：设计数据模式（Schema Design）

**由人工决定，是基础**

#### users.json 模式
```json
{
  "user_id": {
    "name": {
      "first_name": "string",
      "last_name": "string"
    },
    "address": {
      "address1": "string",
      "address2": "string",
      "city": "string",
      "country": "string",
      "state": "string",
      "zip": "string"
    },
    "email": "string",
    "payment_methods": {
      "payment_id": {
        "source": "paypal|credit_card|gift_card",
        "brand": "string (仅credit_card)",
        "last_four": "string (仅credit_card)",
        "balance": "number (仅gift_card)",
        "id": "string"
      }
    },
    "orders": ["order_id_list"]
  }
}
```

**设计考虑**：
- 支持多种支付方式（PayPal、信用卡、礼品卡）
- 支持地址管理（修改地址任务）
- 关联订单列表（快速查找用户订单）

#### products.json 模式
```json
{
  "product_id": {
    "name": "string",
    "product_id": "string",
    "variants": {
      "item_id": {
        "item_id": "string (10位数字)",
        "options": {
          "color": "string",
          "size": "string",
          "material": "string",
          "style": "string",
          "...": "产品特定属性"
        },
        "available": "boolean",
        "price": "number"
      }
    }
  }
}
```

**设计考虑**：
- 每个产品有多个变体（variants）- 颜色、尺寸、材质等组合
- 支持产品搜索和比较
- 支持交换同类产品的不同变体

#### orders.json 模式
```json
{
  "order_id": {
    "order_id": "string (#W开头)",
    "user_id": "string",
    "address": "对象（配送地址）",
    "items": [
      {
        "name": "string",
        "product_id": "string",
        "item_id": "string (10位数字)",
        "price": "number",
        "options": "对象（产品选项）"
      }
    ],
    "fulfillments": [
      {
        "tracking_id": ["string"],
        "item_ids": ["string"]
      }
    ],
    "status": "pending|processed|delivered|cancelled|...",
    "payment_history": [
      {
        "transaction_type": "payment|refund",
        "amount": "number",
        "payment_method_id": "string"
      }
    ]
  }
}
```

**设计考虑**：
- 支持订单状态管理（pending、processed、delivered等）
- 支持多物品订单
- 支持退款和交换
- 记录完整的支付历史

### 阶段2：确定生成方式（Generation Strategy）

**分为两类：程序生成 vs. GPT生成**

#### 需要 GPT 生成的部分（多样性、真实性）

1. **用户名字**
   - `first_name`: "Noah", "Ivan", "Anya", "Sara", "John"
   - `last_name`: "Brown", "Santos", "Garcia", "Doe", "Smith"
   - **原因**：需要真实、多样的人名

2. **地址信息**
   - `address1`: "986 Sunset Drive", "477 Park Avenue"
   - `address2`: "Suite 259", "Apt 12B"
   - `city`: "Denver", "Dallas", "Philadelphia"
   - **原因**：需要真实的美国地址格式

3. **产品类型和名称**
   - "T-Shirt", "Headphones", "Vacuum Cleaner", "Mechanical Keyboard"
   - **原因**：需要多样化的零售产品类别

4. **产品属性值**
   - 颜色："blue", "red", "black", "purple"
   - 材质："cotton", "polyester", "stainless steel"
   - **原因**：需要符合产品类别的真实属性

#### 程序生成的部分（一致性、规则性）

1. **ID 生成**
   ```python
   user_id = f"{first_name.lower()}_{last_name.lower()}_{random.randint(1000, 9999)}"
   order_id = f"#W{random.randint(1000000, 9999999)}"
   product_id = f"{random.randint(1000000000, 9999999999)}"
   item_id = f"{random.randint(1000000000, 9999999999)}"
   payment_id = f"{payment_type}_{random.randint(1000000, 9999999)}"
   ```

2. **价格计算**
   ```python
   base_price = random.uniform(20, 500)
   variant_price = base_price * random.uniform(0.9, 1.1)
   price = round(variant_price, 2)
   ```

3. **订单状态分配**
   ```python
   status = random.choice(["pending", "processed", "delivered"])
   # 根据任务需求分配不同比例
   ```

4. **邮政编码生成**
   ```python
   zip_code = f"{state_code}{random.randint(100, 999)}"
   # 例如：CO州 -> "80279"
   ```

### 阶段3：组合生成（Programmatic Assembly）

**使用程序将 GPT 生成的种子数据与程序生成的数据组合**

#### 生成流程示例

```python
# 伪代码示例
def generate_mock_data():
    # 1. 使用 GPT 生成种子数据
    seed_names = gpt_generate([
        "Generate 100 realistic first names",
        "Generate 100 realistic last names"
    ])
    seed_addresses = gpt_generate([
        "Generate 100 US street addresses",
        "Generate 50 apartment/suite numbers"
    ])
    seed_products = gpt_generate([
        "Generate 50 retail product types",
        "For each product, generate realistic attributes"
    ])

    # 2. 程序化组合生成用户数据
    users = {}
    for i in range(100):
        first = random.choice(seed_names['first'])
        last = random.choice(seed_names['last'])
        user_id = f"{first.lower()}_{last.lower()}_{random.randint(1000, 9999)}"

        users[user_id] = {
            "name": {"first_name": first, "last_name": last},
            "address": random.choice(seed_addresses),
            "email": f"{first.lower()}.{last.lower()}{random.randint(1000, 9999)}@example.com",
            "payment_methods": generate_payment_methods(),
            "orders": []  # 后续填充
        }

    # 3. 程序化生成产品数据
    products = {}
    for product_type in seed_products:
        product_id = str(random.randint(1000000000, 9999999999))

        # 生成产品变体（所有属性组合）
        variants = generate_all_variants(
            product_type,
            product_type['attributes']
        )

        products[product_id] = {
            "name": product_type['name'],
            "product_id": product_id,
            "variants": variants
        }

    # 4. 程序化生成订单数据
    orders = {}
    for user_id in users:
        num_orders = random.randint(1, 5)

        for _ in range(num_orders):
            order_id = f"#W{random.randint(1000000, 9999999)}"

            # 选择随机物品
            items = select_random_items(products, random.randint(1, 5))

            # 生成订单
            orders[order_id] = {
                "order_id": order_id,
                "user_id": user_id,
                "address": users[user_id]["address"],
                "items": items,
                "status": random.choice(["pending", "delivered"]),
                "payment_history": generate_payment_history(items)
            }

            # 更新用户的订单列表
            users[user_id]["orders"].append(order_id)

    return {
        "users": users,
        "products": products,
        "orders": orders
    }
```

## 数据规模和特点

### 数据量估算（基于文件大小）

```bash
# 实际文件大小
users.json:    ~50-100 个用户
products.json: ~30-50 个产品类型，每个产品 10-50 个变体
orders.json:   ~100-200 个订单
```

### 数据特点

1. **关联性完整**
   - 每个订单的 `user_id` 都能在 users.json 中找到
   - 每个订单的 `item_id` 都能在 products.json 的某个 variant 中找到
   - 每个用户的 `orders` 列表与 orders.json 对应

2. **状态多样性**
   - 订单有不同状态（pending、processed、delivered、cancelled）
   - 产品有可用和不可用的
   - 用户有不同的支付方式组合

3. **符合测试需求**
   - 有足够的 pending 订单用于测试取消操作
   - 有足够的 delivered 订单用于测试退货/交换
   - 产品变体足够丰富用于测试产品搜索和比较

## 数据生成的实际应用

### 与 tau_bench Tasks 的对应

```python
from tau_bench.envs.retail.tasks_test import TASKS_TEST

# 示例任务1：取消订单
task1 = TASKS_TEST[0]
# instruction: "我要取消订单#W2378156"
# actions: [
#   Action(name="get_user_details", kwargs={"user_id": "..."}),
#   Action(name="get_order_details", kwargs={"order_id": "#W2378156"}),
#   Action(name="cancel_pending_order", kwargs={...})
# ]
# 数据要求：orders.json 中必须有 order_id="#W2378156" 且 status="pending"

# 示例任务2：交换物品
task2 = TASKS_TEST[n]
# instruction: "我要交换订单中的键盘为带clicky开关的"
# actions: [...product查询, ...exchange动作...]
# 数据要求：
# - orders.json 中有已delivered的订单，包含键盘
# - products.json 中有mechanical keyboard的多个variants（不同switch类型）
```

### 数据一致性保证

为了确保 tau_bench 测试能正确运行，数据生成时必须保证：

1. **ID 唯一性**
   ```python
   all_user_ids = set(users.keys())
   all_order_ids = set(orders.keys())
   # 确保没有重复
   assert len(all_user_ids) == len(users)
   ```

2. **引用完整性**
   ```python
   # 订单中的 user_id 必须存在
   for order in orders.values():
       assert order["user_id"] in users

   # 订单中的 item_id 必须存在于某个产品的variants中
   for item in order["items"]:
       found = False
       for product in products.values():
           if item["item_id"] in product["variants"]:
               found = True
               break
       assert found
   ```

3. **状态合法性**
   ```python
   # pending 订单不能有 tracking_id
   # delivered 订单必须有 tracking_id
   for order in orders.values():
       if order["status"] == "pending":
           assert "fulfillments" not in order or not order["fulfillments"]
       elif order["status"] == "delivered":
           assert "fulfillments" in order and order["fulfillments"]
   ```

## 数据加载机制

### load_data() 函数

```python
# data/__init__.py
def load_data() -> dict[str, Any]:
    """
    加载所有模拟数据库

    Returns:
        {
            "users": {...},
            "products": {...},
            "orders": {...}
        }
    """
    # 使用 OrderedDict 保证键顺序一致性
    # 这对于 hash 计算很重要（tau_bench 奖励计算）
```

### 在测试中的使用

```python
# 在 env.py 中
from data import load_data

# 初始化环境时
self.data_state = DataState(load_data())

# 在奖励计算时
golden_data = load_data()  # 获取原始数据
# 执行 ground truth 动作
# 比较当前状态和 golden 状态
```

## 总结

### 数据生成的关键原则

1. **任务驱动**：数据完全为测试任务设计
2. **混合生成**：GPT生成真实感 + 代码生成一致性
3. **关联完整**：所有引用都有效，无悬挂指针
4. **状态多样**：覆盖各种测试场景
5. **规模适中**：足够测试但不会太大影响性能

### 与 tau_bench 的集成

```
tau_bench.envs.retail.tasks_test.TASKS_TEST
    ↓ 定义测试任务
    ↓
data/ 文件夹
    ↓ 提供支持数据
    ↓
tools/ 模块
    ↓ 提供操作API
    ↓
env.py
    ↓ 运行智能体-环境交互
    ↓
评估结果（reward=0.0 或 1.0）
```

这个设计确保了：
- 测试任务有足够的数据支撑
- 智能体可以正确执行所有必要的操作
- 评估结果准确反映智能体的能力