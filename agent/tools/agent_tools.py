"""AI短剧与AI漫剧提示词优化 Agent 使用的工具。

工具按照用户意图划分，而不是按照知识文件划分。这样扩充知识库时不需要新增工具，
只需要向对应的知识文件补充资料即可。
"""

import os

from langchain_core.tools import tool

from rag.rag_service import RagSummarizeService
from utils.config_handler import chroma_conf
from utils.file_handler import list_dir_with_allowed
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


rag = RagSummarizeService()


def _rag_summarize(query: str, tool_name: str) -> str:
    """统一执行RAG任务，并记录工具内部异常。"""
    try:
        logger.info(f"[{tool_name}]开始处理知识库任务")
        result = rag.rag_summarize(query)
        logger.info(f"[{tool_name}]知识库任务处理完成")
        return result
    except Exception as e:
        logger.error(f"[{tool_name}]知识库任务处理失败，原因：{str(e)}")
        raise e


@tool(
    description=(
        "查询AI短剧与AI漫剧私有知识库。适用于咨询人物设定、脚本结构、美术设计、"
        "分镜、运镜、即梦、Seedance和提示词质量等专业知识。"
        "不要用于用户已经明确要求直接优化提示词、生成完整分镜或进行质量评分的场景。"
    )
)
def query_prompt_knowledge(query: str) -> str:
    """根据提示词知识库回答专业创作问题。"""
    if not query or not query.strip():
        logger.warning("[query_prompt_knowledge]查询内容为空")
        return "查询内容不能为空，请提供需要检索的创作问题。"

    return _rag_summarize(query.strip(), "query_prompt_knowledge")



@tool(
    description=(
        "优化用户已经提供的AI图片或AI视频提示词。适用于润色、补全、增强画面感、"
        "改进运镜、提升人物一致性或适配即梦和Seedance。"
        "用户只有主题而没有原提示词时，不要调用此工具。"
    )
)
def optimize_creation_prompt(
    original_prompt: str,
    platform: str = "通用",
    optimization_goal: str = "综合优化",
) -> str:
    """结合知识库优化用户已经提供的创作提示词。"""
    if not original_prompt or not original_prompt.strip():
        logger.warning("[optimize_creation_prompt]原始提示词为空")
        return "原始提示词不能为空，请提供需要优化的提示词。"

    safe_platform = platform.strip() if platform and platform.strip() else "通用"
    safe_goal = optimization_goal.strip() if optimization_goal and optimization_goal.strip() else "综合优化"

    query = f"""
任务类型：优化现有提示词
目标平台：{safe_platform}
优化目标：{safe_goal}

用户原始提示词：
{original_prompt.strip()}

请结合知识库完成以下任务：
1. 保留用户的主题、人物关系、核心动作和明确限制；
2. 检查主体、动作、场景、镜头、光影、风格、时长和连续性；
3. 删除重复、冲突、空泛或不可执行的表达；
4. 输出一版可以直接复制的优化提示词；
5. 简要说明最重要的修改，不生成大量同构版本。
""".strip()

    return _rag_summarize(query, "optimize_creation_prompt")





@tool(
    description=(
        "把主题、故事梗概或剧情片段转换成结构化视频分镜。适用于生成镜号、时间轴、"
        "画面、动作、景别、运镜、声音和连续性要求。"
        "只咨询分镜概念时使用知识库查询工具，不要调用此工具。"
    )
)
def generate_storyboard(
    theme: str,
    duration: int = 15,
    platform: str = "通用",
    aspect_ratio: str = "9:16",
    style: str = "未指定",
) -> str:
    """结合知识库将创作主题转换为可执行分镜。"""
    if not theme or not theme.strip():
        logger.warning("[generate_storyboard]创作主题为空")
        return "创作主题不能为空，请提供主题、故事梗概或剧情片段。"

    safe_duration = max(5, min(duration, 300))
    if safe_duration != duration:
        logger.warning(
            f"[generate_storyboard]用户时长{duration}秒超出范围，已调整为{safe_duration}秒"
        )

    safe_platform = platform.strip() if platform and platform.strip() else "通用"
    safe_aspect_ratio = aspect_ratio.strip() if aspect_ratio and aspect_ratio.strip() else "9:16"
    safe_style = style.strip() if style and style.strip() else "未指定"

    query = f"""
任务类型：生成AI短剧或AI漫剧分镜
创作主题：{theme.strip()}
目标时长：{safe_duration}秒
目标平台：{safe_platform}
画面比例：{safe_aspect_ratio}
视觉风格：{safe_style}

请结合知识库生成结构化分镜：
1. 先给出一句话故事和必要的创意假设；
2. 建立主要角色的一致性锚点；
3. 按时间顺序输出镜号、时间段、景别、画面、人物动作、运镜、声音或对白；
4. 所有时间段必须连续、不重叠，总时长等于{safe_duration}秒；
5. 每个镜头只承担一个主要叙事任务；
6. 最后给出人物、道具、场景和动作的连续性要求；
7. 如果指定平台，使用适合该平台的自然语言表达，不编造隐藏参数。
""".strip()

    return _rag_summarize(query, "generate_storyboard")



@tool(
    description=(
        "诊断和评价用户提供的提示词，适用于查找人物漂移、动作混乱、镜头无目的、"
        "风格冲突、时长不合理和表达重复等问题。"
        "用户明确要求直接得到优化成品时，使用提示词优化工具。"
    )
)
def evaluate_prompt_quality(
    prompt: str,
    platform: str = "通用",
    focus: str = "综合质量",
) -> str:
    """结合知识库诊断提示词质量并提出修改建议。"""
    if not prompt or not prompt.strip():
        logger.warning("[evaluate_prompt_quality]待检查提示词为空")
        return "待检查的提示词不能为空。"

    safe_platform = platform.strip() if platform and platform.strip() else "通用"
    safe_focus = focus.strip() if focus and focus.strip() else "综合质量"

    query = f"""
任务类型：提示词质量诊断
目标平台：{safe_platform}
检查重点：{safe_focus}

待检查提示词：
{prompt.strip()}

请结合知识库进行诊断：
1. 从意图保真、主体明确、动作可见、场景完整、镜头明确、视觉统一、连续性、时长匹配、表达效率和平台适配十个维度检查；
2. 每个维度按0至2分评价，并说明扣分原因；
3. 给出总分和三个最优先修改的问题；
4. 提供具体修改建议，但不要把没有依据的随机差异当作质量结论。
""".strip()

    return _rag_summarize(query, "evaluate_prompt_quality")


@tool(
    description=(
        "返回系统当前支持的创作能力、平台和知识库范围。适用于用户询问系统能做什么、"
        "支持哪些提示词类型、是否支持图片或知识库包含哪些资料。这个工具不需要参数。"
    )
)
def list_supported_capabilities() -> str:
    """读取知识文件列表并返回当前系统能力边界。"""
    knowledge_path = get_abs_path(chroma_conf["data_path"])
    knowledge_files = list_dir_with_allowed(knowledge_path, (".txt",))

    if not knowledge_files:
        logger.error(f"[list_supported_capabilities]没有找到知识文件：{knowledge_path}")
        return "当前没有找到可用的提示词知识文件。"

    file_names = [os.path.basename(file_path) for file_path in knowledge_files]
    file_text = "\n".join(
        f"{index}. {file_name}" for index, file_name in enumerate(file_names, start=1)
    )

    return (
        "当前系统支持以下能力：\n"
        "1. AI短剧与AI漫剧提示词优化\n"
        "2. 从主题扩展人物、剧情、视觉和提示词\n"
        "3. 人物一致性、脚本、美术、分镜和运镜知识咨询\n"
        "4. 生成指定时长的视频分镜和时间轴\n"
        "5. 即梦与Seedance文字提示词适配\n"
        "6. 提示词质量诊断与修改建议\n\n"
        "当前只处理文字输入，不分析图片、视频或音频。\n"
        "知识库文件：\n"
        f"{file_text}"
    )


@tool(
    description=(
        "无入参。仅当用户明确要求完整策划案、详细分镜方案、完整制作方案或系统化创作报告时调用。"
        "调用后由中间件切换到完整创作方案提示词。普通优化、普通分镜和知识咨询严禁调用。"
    )
)
def fill_context_for_creation_plan() -> str:
    """触发中间件进入完整创作方案模式。"""
    logger.info("[fill_context_for_creation_plan]请求切换到完整创作方案模式")
    return "已进入完整创作方案模式。"


# Agent创建时直接导入这个列表，避免定义了工具却忘记注册。
PROMPT_TOOLS = [
    query_prompt_knowledge,
    optimize_creation_prompt,
    generate_storyboard,
    evaluate_prompt_quality,
    list_supported_capabilities,
    fill_context_for_creation_plan,
]


# 临时兼容react_agent.py中的旧导入名，后续修改该文件时删除此别名。
CAREER_TOOLS = PROMPT_TOOLS
