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
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
import yaml

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


def load_test_cases():
    """读取RAG测试集，并根据case_limit选择参与评测的用例数量。"""
    test_cases_path = get_abs_path(eval_conf["dataset_path"])
    with open(test_cases_path, "r", encoding="utf-8") as f:
        test_cases_data = yaml.load(f, Loader=yaml.FullLoader)

    test_cases = test_cases_data["cases"]
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


def print_rag_evaluation_results(test_results, test_cases_data):
    """输出每条用例的分数，以及各指标和全部分数的平均值。"""
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
    print("平均分汇总")
    print("=" * 70)

    for metric_name, scores in metric_scores.items():
        average_score = sum(scores) / len(scores)
        print(f"{metric_name}: {average_score:.4f}")

    if all_scores:
        overall_average = sum(all_scores) / len(all_scores)
        print("-" * 70)
        print(f"五项指标总体平均分: {overall_average:.4f}")


def test_rag_quality():
    # 第一步：从YAML测试集中读取全部用例。
    test_cases_data = load_test_cases()

    # 第二步：复用一个RAG服务，逐条完成真实检索和回答生成。
    rag_service = RagSummarizeService()
    rag_test_cases = []

    for case_data in test_cases_data:
        query = case_data["input"]
        retrieved_docs = rag_service.retriever_docs(query)
        actual_output = rag_service.rag_summarize(query)

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

    # 第四步：将10条用例一次性交给DeepEval官方evaluate函数。
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
