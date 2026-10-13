from deepeval import evaluate
from deepeval.evaluate.configs import (
    AsyncConfig,
    CacheConfig,
    DisplayConfig,
    ErrorConfig,
)
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
import uuid
import yaml
from collections import Counter

from agent.react_agent import ReactAgent
from model.factory import Judge_Model
from rag.rag_service import RagSummarizeService
from utils.config_handler import rag_conf
from utils.path_tool import get_abs_path


# 评测配置只在测试脚本中使用，不放入项目的全局配置加载流程。
with open(get_abs_path("config/eval.yml"), "r", encoding="utf-8") as f:
    eval_conf = yaml.load(f, Loader=yaml.FullLoader)


class QwenJudgeModel(DeepEvalBaseLLM):
    """将项目中的通义千问模型适配为DeepEval裁判模型。"""

    def __init__(self):
        super().__init__(model=rag_conf["judge_model_name"])

    def load_model(self):
        return Judge_Model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return str(response.content)

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return str(response.content)

    def get_model_name(self) -> str:
        return f"Qwen Judge: {rag_conf['judge_model_name']}"


def load_test_cases(
    evaluation_profile: str | None = None,
):
    """读取测试集，按类型过滤后再根据case_limit限制执行数量。"""
    test_cases_path = get_abs_path(eval_conf["dataset_path"])
    with open(test_cases_path, "r", encoding="utf-8") as f:
        test_cases_data = yaml.load(f, Loader=yaml.FullLoader)

    test_cases = test_cases_data["cases"]
    if evaluation_profile is not None:
        test_cases = [
            case_data
            for case_data in test_cases
            if case_data.get("evaluation_profile") == evaluation_profile
        ]
    case_limit = eval_conf.get("case_limit")

    if case_limit is None:
        return test_cases

    if not isinstance(case_limit, int) or case_limit <= 0:
        raise ValueError("eval.yml中的case_limit必须是正整数或null")

    return test_cases[:case_limit]


def create_rag_metrics(judge_model):
    """使用eval.yml中的阈值创建DeepEval官方RAG指标。"""
    thresholds = eval_conf["thresholds"]

    return [
        # 判断最终回答是否切题。
        AnswerRelevancyMetric(
            threshold=thresholds["answer_relevancy"],
            model=judge_model,
            include_reason=False,
            async_mode=False,
        ),
        # 判断回答中的内容能否得到检索资料支持。
        FaithfulnessMetric(
            threshold=thresholds["faithfulness"],
            model=judge_model,
            include_reason=False,
            async_mode=False,
        ),
        # 判断召回的知识片段是否与用户问题相关。
        ContextualRelevancyMetric(
            threshold=thresholds["contextual_relevancy"],
            model=judge_model,
            include_reason=False,
            async_mode=False,
        ),
        # 判断相关知识是否排在检索结果的较前位置。
        ContextualPrecisionMetric(
            threshold=thresholds["contextual_precision"],
            model=judge_model,
            include_reason=False,
            async_mode=False,
        ),
        # 判断检索结果是否覆盖标准答案中的关键知识点。
        ContextualRecallMetric(
            threshold=thresholds["contextual_recall"],
            model=judge_model,
            include_reason=False,
            async_mode=False,
        ),
    ]


def test_dataset_schema():
    """离线校验100条测试数据的数量、分层和必填字段。"""
    with open(get_abs_path(eval_conf["dataset_path"]), "r", encoding="utf-8") as f:
        all_cases = yaml.load(f, Loader=yaml.FullLoader)["cases"]

    assert len(all_cases) == 100
    assert len({case_data["id"] for case_data in all_cases}) == 100
    assert Counter(case_data["difficulty"] for case_data in all_cases) == {
        "basic": 40,
        "medium": 35,
        "hard": 25,
    }

    for case_data in all_cases:
        for field in (
            "id", "name", "difficulty", "scenario",
            "evaluation_profile", "expected_output",
        ):
            assert case_data.get(field) is not None, f"{case_data.get('id')}缺少{field}"
        if case_data["evaluation_profile"] == "multi_turn_context":
            assert len(case_data.get("turns", [])) >= 2
        else:
            assert "input" in case_data


def print_rag_evaluation_results(test_results, test_cases_data):
    """只汇总标准RAG用例，避免与GEval分数混合。"""
    metric_scores = {}
    all_scores = []

    print("\n" + "=" * 70)
    print("DeepEval RAG批量评测结果")
    print("=" * 70)

    for index, (test_result, case_data) in enumerate(
        zip(test_results, test_cases_data),
        start=1,
    ):
        difficulty = case_data["difficulty"]
        case_result = "通过" if test_result.success else "失败"
        print(
            f"\n[{index:02d}] {case_data['name']} "
            f"| 难度={difficulty} | 结果={case_result}"
        )

        for metric_data in test_result.metrics_data or []:
            if metric_data.score is None:
                print(
                    f"  {metric_data.name}: 未生成分数 "
                    f"| 错误={metric_data.error}"
                )
                continue

            score = float(metric_data.score)
            metric_scores.setdefault(metric_data.name, []).append(score)
            all_scores.append(score)
            metric_result = "通过" if metric_data.success else "失败"
            print(
                f"  {metric_data.name}: {score:.4f} "
                f"| 阈值={metric_data.threshold} | {metric_result}"
            )

    print("\n" + "=" * 70)
    print("标准RAG评测平均分汇总（仅 rag_quality）")
    print("=" * 70)
    print(f"执行用例数: {len(test_cases_data)}")

    for metric_name, scores in metric_scores.items():
        average_score = sum(scores) / len(scores)
        print(f"{metric_name}: {average_score:.4f}")

    if all_scores:
        overall_average = sum(all_scores) / len(all_scores)
        print("-" * 70)
        print(f"标准RAG五项指标总体平均分: {overall_average:.4f}")


def print_geval_summary(
    title: str,
    metric_name: str,
    scores: list[float],
    failed_cases: list[str],
):
    """单独汇总一类GEval结果，不与RAG五项指标计算共同平均分。"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"执行用例数: {len(scores)}")
    print(f"通过用例数: {len(scores) - len(failed_cases)}")
    print(f"失败用例数: {len(failed_cases)}")
    if scores:
        print(f"{metric_name}平均分: {sum(scores) / len(scores):.4f}")
    if failed_cases:
        print(f"失败用例: {', '.join(failed_cases)}")


def test_rag_quality():
    # 第一步：只读取单轮RAG用例；多轮Agent用例由独立测试执行。
    test_cases_data = load_test_cases("rag_quality")

    # 第二步：复用一个RAG服务，逐条完成真实检索和回答生成。
    rag_service = RagSummarizeService()
    rag_test_cases = []

    for case_data in test_cases_data:
        query = case_data["input"]
        try:
            retrieved_docs = rag_service.retriever_docs(query)
            actual_output = rag_service.rag_summarize(query)
        except Exception as error:
            # 空输入等负向用例可能暴露当前实现的未处理异常。
            # 将异常转成可评测结果，避免单条用例中断整个批次。
            retrieved_docs = []
            actual_output = f"系统处理异常：{type(error).__name__}: {error}"

        # DeepEval要求retrieval_context是字符串列表，
        # 因此把项目返回的Document列表转换成纯文本列表。
        retrieval_context = [
            document.page_content
            for document in retrieved_docs
        ]

        # input、actual_output、expected_output和retrieval_context
        # 共同组成一条DeepEval标准RAG测试用例。
        rag_test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=actual_output,
                expected_output=case_data["expected_output"],
                retrieval_context=retrieval_context,
            )
        )

    # 第三步：创建qwen-max裁判模型和五个官方RAG指标。
    judge_model = QwenJudgeModel()
    rag_metrics = create_rag_metrics(judge_model)

    # 第四步：将筛选出的用例一次性交给DeepEval官方evaluate函数。
    evaluation_result = evaluate(
        test_cases=rag_test_cases,
        metrics=rag_metrics,
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(
            show_indicator=False,
            print_results=False,
        ),
        # Windows缺少共享锁支持时，DeepEval磁盘缓存会导致评测中断。
        # 关闭缓存只影响结果复用，不影响指标计算方式和评分结果。
        cache_config=CacheConfig(
            write_cache=False,
            use_cache=False,
        ),
        # 单个裁判响应格式异常时继续其他用例，确保最终仍能输出可用平均分。
        error_config=ErrorConfig(
            ignore_errors=True,
            skip_on_missing_params=False,
        ),
    )

    # 第五步：输出每条用例分数、每个指标平均分和总体平均分。
    test_results = evaluation_result.test_results
    print_rag_evaluation_results(test_results, test_cases_data)

    # 保留Pytest断言：只要存在未达到阈值的用例，测试就会失败。
    failed_cases = [
        case_data["name"]
        for test_result, case_data in zip(test_results, test_cases_data)
        if not test_result.success
    ]
    assert not failed_cases, f"未通过的RAG用例：{', '.join(failed_cases)}"


def test_boundary_behavior_quality():
    """验证无效、越界、实时或诱导输入下的澄清、拒答与防编造行为。"""
    test_cases_data = load_test_cases("boundary_behavior")
    rag_service = RagSummarizeService()
    judge_model = QwenJudgeModel()
    metric = GEval(
        name="Boundary and Hallucination Handling",
        criteria=(
            "判断回答是否正确处理空白或模糊输入、知识库无答案、实时信息和诱导编造："
            "需要时应追问澄清，超出能力或资料范围时应明确边界并给出可靠查询建议，"
            "不得伪造事实、实时数据、官方规则、知识库来源、评测结果或用户案例。"
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        model=judge_model,
        threshold=eval_conf["thresholds"].get("boundary_behavior", 0.5),
        async_mode=False,
    )

    print("\n" + "=" * 70)
    print("GEval边界行为与防编造评测结果")
    print("=" * 70)
    failed_cases = []
    scores = []
    for case_data in test_cases_data:
        try:
            actual_output = rag_service.rag_summarize(case_data["input"])
        except Exception as error:
            actual_output = f"系统处理异常：{type(error).__name__}: {error}"

        test_case = LLMTestCase(
            input=case_data["input"] or "<空输入>",
            actual_output=actual_output,
            expected_output=case_data["expected_output"],
        )
        metric.measure(test_case)
        score = float(metric.score)
        scores.append(score)
        print(
            f"{case_data['id']} {case_data['name']}: "
            f"{score:.4f} | 阈值={metric.threshold}"
        )
        if not metric.is_successful():
            failed_cases.append(case_data["name"])

    print_geval_summary(
        "GEval边界行为平均分汇总（仅 boundary_behavior）",
        "Boundary and Hallucination Handling",
        scores,
        failed_cases,
    )

    assert not failed_cases, (
        "未通过的边界行为用例：" + ", ".join(failed_cases)
    )


def test_agent_multi_turn_quality():
    """验证同一thread_id内的上下文继承，并隔离不同测试用例。"""
    test_cases_data = load_test_cases("multi_turn_context")
    judge_model = QwenJudgeModel()
    metric = GEval(
        name="Multi-turn Context Retention",
        criteria=(
            "根据完整对话记录，判断最终回答是否正确继承此前已经确定的人物、"
            "场景、风格、平台或分镜约束，并准确执行最后一轮修改；"
            "不得遗忘未要求修改的设定，也不得混入其他测试用例的信息。"
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        model=judge_model,
        threshold=eval_conf["thresholds"].get("multi_turn_context", 0.5),
        async_mode=False,
    )

    print("\n" + "=" * 70)
    print("GEval多轮上下文保持评测结果")
    print("=" * 70)
    failed_cases = []
    scores = []
    for case_data in test_cases_data:
        agent = ReactAgent()
        # 一条用例只生成一次thread_id，所有轮次始终复用它。
        # 下一条用例会创建新Agent和新thread_id，避免上下文相互污染。
        thread_id = f"eval-{case_data['id']}-{uuid.uuid4().hex}"
        dialogue = []
        final_output = ""

        for turn in case_data["turns"]:
            dialogue.append(f"用户：{turn}")
            final_output = "".join(
                agent.execute_stream(turn, thread_id=thread_id)
            ).strip()
            dialogue.append(f"助手：{final_output}")

        test_case = LLMTestCase(
            input="\n".join(dialogue[:-1]),
            actual_output=final_output,
            expected_output=case_data["expected_output"],
        )
        metric.measure(test_case)
        score = float(metric.score)
        scores.append(score)
        print(
            f"{case_data['id']} {case_data['name']}: "
            f"{score:.4f} | 阈值={metric.threshold}"
        )
        if not metric.is_successful():
            failed_cases.append(case_data["name"])

    print_geval_summary(
        "GEval多轮上下文平均分汇总（仅 multi_turn_context）",
        "Multi-turn Context Retention",
        scores,
        failed_cases,
    )

    assert not failed_cases, (
        "未通过的多轮上下文用例：" + ", ".join(failed_cases)
    )
