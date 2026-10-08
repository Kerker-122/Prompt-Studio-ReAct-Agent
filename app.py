import uuid

import streamlit as st

from agent.react_agent import ReactAgent
from utils.logger_handler import logger


st.set_page_config(
    page_title="镜语 Prompt Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 78% 8%, rgba(124, 92, 255, 0.12), transparent 28rem),
            radial-gradient(circle at 8% 88%, rgba(33, 195, 166, 0.09), transparent 24rem);
    }
    .block-container {
        max-width: 1100px;
        padding-top: 2.2rem;
        padding-bottom: 5rem;
    }
    .hero {
        padding: 1.65rem 1.8rem;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(124, 92, 255, 0.22);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(124, 92, 255, 0.11), rgba(33, 195, 166, 0.07));
    }
    .hero-tag {
        margin-bottom: 0.55rem;
        color: #8b7cff;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.05rem;
        line-height: 1.2;
    }
    .hero p {
        margin: 0.65rem 0 0;
        color: rgba(128, 128, 128, 0.95);
        line-height: 1.7;
    }
    .feature-card {
        min-height: 126px;
        padding: 1rem 1.05rem;
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 16px;
        background: rgba(128, 128, 128, 0.04);
    }
    .feature-card strong {
        display: block;
        margin-bottom: 0.38rem;
        font-size: 1rem;
    }
    .feature-card span {
        color: rgba(128, 128, 128, 0.95);
        font-size: 0.88rem;
        line-height: 1.55;
    }
    [data-testid="stChatMessage"] {
        padding: 0.35rem 0;
    }
    [data-testid="stCode"] pre,
    [data-testid="stCode"] code {
        white-space: pre-wrap !important;
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
    }
    [data-testid="stCode"] pre {
        overflow-x: hidden !important;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.13);
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        border: 0;
        background: linear-gradient(135deg, #7257ff, #20bca0);
        color: white;
    }
    .session-label {
        color: rgba(128, 128, 128, 0.9);
        font-size: 0.78rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = uuid.uuid4().hex


selected_prompt = None

with st.sidebar:
    st.markdown("## 🎬 镜语")
    st.caption("AI短剧与AI漫剧提示词优化 Agent")

    if st.button("＋ 新建对话", type="primary", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["thread_id"] = uuid.uuid4().hex
        st.rerun()

    st.divider()
    st.subheader("创作设置")

    platform = st.selectbox(
        "目标平台",
        ["通用", "即梦", "Seedance"],
        help="未确定平台时选择通用，Agent不会使用未经确认的平台专属参数。",
    )

    work_mode = st.selectbox(
        "工作模式",
        ["智能判断", "快速优化", "完整创作方案"],
        help="智能判断会根据问题自动选择工具；完整创作方案会启用动态提示词。",
    )

    st.divider()
    st.subheader("试试这些问题")

    example_prompts = [
        "帮我把这段提示词改得更有画面感：雨夜，一个剑客独自走进破庙。",
        "设计一个15秒Seedance分镜：失忆剑客发现自己的佩剑会说话。",
        "AI漫剧怎样保持同一人物在多个镜头中的外貌一致？",
        "为‘记忆可以交易’设计一份完整的AI短剧创作方案。",
    ]

    for index, example in enumerate(example_prompts):
        if st.button(example, key=f"example_{index}", use_container_width=True):
            selected_prompt = example

    st.divider()
    st.markdown("**当前能力**")
    st.caption("提示词优化 · 主题扩展 · 分镜生成 · 质量诊断 · 即梦 / Seedance适配")
    st.markdown(
        f'<div class="session-label">会话 ID：{st.session_state["thread_id"][:8]}</div>',
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <section class="hero">
        <div class="hero-tag">PROMPT KNOWLEDGE · RAG · REACT AGENT</div>
        <h1>把一个灵感，变成可执行的镜头语言</h1>
        <p>输入主题、故事或现有提示词，获得人物一致性设计、剧情扩展、分镜运镜与平台适配建议。</p>
    </section>
    """,
    unsafe_allow_html=True,
)


if not st.session_state["messages"]:
    card_columns = st.columns(3)
    card_contents = [
        ("✦ 快速优化", "保留原始创意，补足主体、动作、镜头、光影和连续性。"),
        ("▦ 分镜生成", "将主题或故事转换成带时间轴、景别和运镜的可执行方案。"),
        ("◎ 知识检索", "从私有知识库查询人物、脚本、美术、分镜和平台提示词方法。"),
    ]

    for column, (title, description) in zip(card_columns, card_contents):
        with column:
            st.markdown(
                f'<div class="feature-card"><strong>{title}</strong><span>{description}</span></div>',
                unsafe_allow_html=True,
            )

    st.info("可以直接输入一句创意；如果你已经有提示词，粘贴后说明希望优化的方向即可。")


for message in st.session_state["messages"]:
    avatar = "🧑‍💻" if message["role"] == "user" else "🎬"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


prompt = st.chat_input("输入主题、故事、提示词，或询问分镜与运镜知识……")
if selected_prompt:
    prompt = selected_prompt


if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    agent_query_parts = []
    if platform != "通用":
        agent_query_parts.append(f"目标平台：{platform}")

    if work_mode == "快速优化":
        agent_query_parts.append("工作模式：请按已有提示词快速优化模式处理")
    elif work_mode == "完整创作方案":
        agent_query_parts.append("工作模式：请生成完整创作方案，并先切换到完整创作方案模式")

    agent_query_parts.append(f"用户需求：{prompt}")
    agent_query = "\n".join(agent_query_parts)

    response_chunks: list[str] = []
    response_failed = False

    def stream_response():
        for chunk in st.session_state["agent"].execute_stream(
            agent_query,
            thread_id=st.session_state["thread_id"],
        ):
            response_chunks.append(chunk)
            yield chunk

    with st.chat_message("assistant", avatar="🎬"):
        try:
            with st.spinner("正在理解创作需求并查询知识库……"):
                st.write_stream(stream_response())
        except Exception as error:
            response_failed = True
            logger.error(f"[Streamlit]回答生成失败，原因：{str(error)}", exc_info=True)
            st.error("回答生成失败，请检查网络、模型配置和日志后重试。")

    response = "".join(response_chunks).strip()
    if response and not response_failed:
        st.session_state["messages"].append(
            {"role": "assistant", "content": response}
        )
        st.rerun()
