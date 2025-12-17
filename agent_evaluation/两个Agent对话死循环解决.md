# 对话死循环修复说明

## 问题描述

在智能体与用户模拟器的对话中，会出现"礼貌性死循环"现象：

```
Agent: You're welcome! Have a great day! 😊
User: Thank you, you too! Have a great day. 😊
Agent: You're welcome! Have a great day! 😊
User: Thank you, you too! Have a great day. 😊
...（无限循环）
```

这种情况发生在任务已经完成，但用户模拟器没有发送`###STOP###`信号，而是继续礼貌性回复。

## 根本原因

1. **用户模拟器未发送停止信号**：任务完成后，用户模拟器应该发送`###STOP###`，但它继续生成礼貌性回复
2. **缺少重复检测机制**：代码没有检测到对话陷入重复循环
3. **系统提示词不够明确**：用户模拟器的提示词没有足够强调何时应该终止对话

## 修复方案

### 1. 添加对话循环检测 (env.py:199-229)

新增 `_detect_conversation_loop()` 方法来检测重复消息：

```python
def _detect_conversation_loop(self, recent_messages, threshold=3):
    """
    检测对话是否陷入重复循环

    - 比较最近的消息（忽略表情符号和标点）
    - 如果连续出现相同或高度相似的消息，判定为循环
    - 默认阈值：连续3次相似消息
    """
```

**工作原理**：
- 规范化消息内容（移除表情符号、标点）
- 比较最近的消息是否高度相似
- 如果相似度超过阈值，返回True

### 2. 添加礼貌性结束检测 (env.py:231-263)

新增 `_detect_polite_closure()` 方法来识别礼貌性结束语：

```python
def _detect_polite_closure(self, agent_msg, user_msg):
    """
    检测对话是否进入礼貌性结束阶段

    - 检查双方是否都使用了结束语关键词
    - 关键词包括："have a great day", "thank you", "goodbye"等
    - 如果双方消息都很短且包含结束语，判定为礼貌性结束
    """
```

**检测关键词**：
- "have a great day"
- "have a nice day"
- "you're welcome"
- "thank you"
- "thanks"
- "goodbye"
- "bye"
- "take care"

### 3. 在主循环中应用检测 (env.py:387-410)

在每轮对话后进行检测：

```python
# 检测对话循环
if self._detect_conversation_loop(recent_messages):
    print("\n[检测到对话循环，自动结束对话]")
    done = True

# 检测礼貌性结束
if self._detect_polite_closure(agent_output, user_message):
    polite_closure_count += 1
    print(f"\n[检测到礼貌性结束 {polite_closure_count}/2]")
    if polite_closure_count >= 2:
        print("\n[连续礼貌性结束，自动结束对话]")
        done = True
else:
    polite_closure_count = 0  # 重置计数器
```

**工作流程**：
1. 每轮对话后，将消息添加到 `recent_messages` 列表
2. 检测是否陷入循环（重复消息）
3. 检测是否进入礼貌性结束
4. 如果连续2次礼貌性结束，自动终止对话

### 4. 改进用户模拟器提示词 (env.py:190-198)

增强系统提示词，更明确地指导用户模拟器何时应该发送停止信号：

**新增规则**：
```
- IMPORTANT: If the instruction goal is satisfied (the agent has completed the task),
  you MUST immediately generate '###STOP###' as a standalone message without anything
  else to end the conversation. Do not continue with polite responses like "thank you"
  or "have a great day" after the goal is achieved.

- Do NOT enter into polite back-and-forth exchanges. Once the task is done and you
  say thank you, generate '###STOP###' on the next turn.
```

## 修复效果

### 修复前：
```
Agent: You're welcome! Have a great day! 😊
User: Thank you, you too! Have a great day. 😊
Agent: You're welcome! Have a great day! 😊
User: Thank you, you too! Have a great day. 😊
...（无限循环，达到max_num_steps才停止）
```

### 修复后：
```
Agent: You're welcome! Have a great day! 😊
User: Thank you, you too! Have a great day. 😊
[检测到礼貌性结束 1/2]
Agent: You're welcome! Have a great day! 😊
User: Thank you, you too! Have a great day. 😊
[检测到礼貌性结束 2/2]
[连续礼貌性结束，自动结束对话]
对话正常结束
```

或者：
```
Agent: The task is complete! Have a great day! 😊
User: ###STOP###
对话正常结束（用户模拟器主动发送停止信号）
```

## 技术细节

### 检测参数调优

如需调整检测灵敏度，可修改以下参数：

1. **循环检测阈值** (env.py:388)
   ```python
   if self._detect_conversation_loop(recent_messages, threshold=3):
   ```
   - `threshold=3`: 连续3次相似消息判定为循环
   - 可根据实际情况调整（2-5之间合适）

2. **礼貌性结束计数** (env.py:396)
   ```python
   if polite_closure_count >= 2:
   ```
   - `>= 2`: 连续2次礼貌性结束才终止
   - 避免误判（任务未完成就提前终止）

3. **消息长度阈值** (env.py:263)
   ```python
   len(agent_msg) < 50 and len(user_msg) < 50
   ```
   - 只有短消息（< 50字符）才判定为礼貌性结束
   - 避免误判正常的长回复

## 其他改进建议

如果仍然出现死循环，可以考虑：

1. **降低 max_num_steps**：将默认值从30降低到15-20
2. **使用更好的用户模拟器模型**：某些模型更善于遵循停止指令
3. **添加强制超时**：如果最后N轮没有工具调用，自动终止
4. **在智能体提示词中强调**：让智能体在任务完成后明确说"Task completed"

## 测试建议

运行测试时，观察以下输出：
- `[检测到对话循环，自动结束对话]` - 说明循环检测工作正常
- `[检测到礼貌性结束 X/2]` - 说明礼貌性检测工作正常
- `###STOP###` - 说明用户模拟器正确发送了停止信号

## 文件修改清单

- ✅ `env.py`: 添加循环检测和礼貌性结束检测功能
- ✅ `CONVERSATION_LOOP_FIX.md`: 本文档（修复说明）

## 相关文件

如果使用 `env_litellm.py`，需要应用相同的修改。