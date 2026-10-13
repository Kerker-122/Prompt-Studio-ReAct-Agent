# 100条分层评测集说明

## 数据分布

| 难度 | 数量 | 主要场景 |
| --- | ---: | --- |
| 基础 | 40 | 简单知识点检索、提示词补全 |
| 中等 | 35 | 多条件、多知识点、多轮上下文、冲突约束 |
| 困难 | 25 | 空输入、模糊问题、无答案、实时信息、诱导编造 |

测试集共100条，其中92条为单轮RAG用例，8条为多轮Agent用例。`scenario`表示业务场景，`evaluation_profile`决定使用哪类评测方法：

- `rag_quality`：使用DeepEval五项RAG指标评估有明确知识答案的用例。
- `boundary_behavior`：使用GEval评估澄清、拒答、知识边界和防编造行为，不使用不适合拒答场景的Contextual Recall。
- `multi_turn_context`：使用GEval比较完整对话、最终回答和期望结果，评估上下文继承。

三套评价体系独立输出统计结果：

- `rag_quality`输出五项指标各自平均分，以及仅在标准RAG用例范围内计算的五项总体平均分。
- `boundary_behavior`输出Boundary and Hallucination Handling单项分数、通过/失败数和独立平均分。
- `multi_turn_context`输出Multi-turn Context Retention单项分数、通过/失败数和独立平均分。

不同体系评价的能力与分数含义不同，因此不计算100条用例的跨体系总平均分。

## 各问题类型的两个代表用例

| 验证类型 | 用例一 | 用例二 | 主要验证点 |
| --- | --- | --- | --- |
| 简单知识点检索 | B001 人物固定锚点 | B004 分镜必要字段 | 是否召回正确知识并准确回答 |
| 提示词补全 | B021 雨夜女剑客补全 | B038 竹林对决补全 | 是否补齐主体、动作、镜头、光影和连续性 |
| 多条件 | M041 竖屏雨夜追凶 | M046 一镜到底古城 | 是否同时满足时长、比例、风格、人物及镜头条件 |
| 多知识点 | M051 角色卡到分镜 | M053 主题转完整方案 | 是否组合人物、剧情、视觉、分镜和运镜知识 |
| 多轮上下文 | M060 多轮修改主角性别 | M067 多轮否定上轮修改 | 是否记住既有条件并仅执行指定修改或撤销 |
| 冲突约束 | M068 固定机位与环绕冲突 | M074 时长总和冲突 | 是否识别不可同时满足的条件并澄清或修正 |
| 空输入 | H076 空字符串输入 | H079 只有标点输入 | 是否避免无依据生成并引导补充需求 |
| 模糊问题 | H080 模糊优化请求 | H083 不明确的平台适配 | 是否识别缺失信息并提出有效澄清问题 |
| 知识库无答案 | H085 医学诊断越界 | H088 平台内部算法越界 | 是否声明能力边界并避免生成专业或内部结论 |
| 实时信息 | H090 Seedance最新价格 | H092 百炼实时限流 | 是否避免把非实时知识库当成最新事实来源 |
| 诱导编造 | H095 要求虚构官方参数 | H098 伪造评测高分 | 是否拒绝伪造参数、来源和测试结果 |

## 多轮上下文如何保证thread_id相同

`test_agent_multi_turn_quality`对每条多轮用例只生成一次`thread_id`，随后将该用例的所有`turns`依次传给同一个Agent：

```python
thread_id = f"eval-{case_data['id']}-{uuid.uuid4().hex}"

for turn in case_data["turns"]:
    final_output = "".join(
        agent.execute_stream(turn, thread_id=thread_id)
    )
```

这能保证一条用例内的所有轮次命中LangGraph同一个检查点。每进入下一条用例，测试会重新创建Agent并生成新的UUID，从而避免上一条用例的记忆污染下一条。

## 执行方式

开发阶段默认`case_limit: 10`，以控制在线模型调用量。将其改为`null`才会执行全部用例。

```powershell
# 只做离线的数据数量、分层和字段校验
python -m pytest -q agent_test.py -k dataset_schema

# 标准RAG五指标评测
python -m pytest -q agent_test.py -k rag_quality -s

# 空输入、越界、实时信息和诱导编造评测
python -m pytest -q agent_test.py -k boundary_behavior_quality -s

# 多轮上下文评测
python -m pytest -q agent_test.py -k agent_multi_turn_quality -s
```

100条用例的全量结果必须在实际运行后重新统计，不能沿用扩充前10条基线用例的0.8207总体平均分。
