from utils.logger_handler import logger
from utils.prompt_loader import load_sys_prompts,load_report_prompts
from typing import Callable
from langchain.agents.middleware import (
    AgentState,
    ModelRequest,
    before_model,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command


@wrap_tool_call
def monitor_tool(               #工具执行监控
    #请求的数据封装
    request:ToolCallRequest,
    #执行函数本身
    handle:Callable[[ToolCallRequest],ToolMessage | Command],
)->ToolMessage | Command:  
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]执行工具：{request.tool_call['args']}")

    try:
        result=handle(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        if request.tool_call['name']=="fill_context_for_creation_plan":#tools拿不到上下文，所以由中间件记录完整创作方案状态
            request.runtime.context["creation_plan"]=True
        return result
    except Exception as e:
        logger.error(f"[tool monitor]工具{request.tool_call['name']}调用失败，原因{str(e)}")
        raise e

@before_model
def log_before_model(           #模型执行前输出日志
    state:AgentState,           #整个智能体中的状态节点
    runtime:Runtime,            #记录执行过程中的上下文信息
):
    messages = state["messages"]
    logger.info(f"[log_before_model]即将调用模型，带有{len(messages)}条消息")
    if messages:
        logger.debug(
            f"[log_before_model]{type(messages[-1]).__name__} | "
            f"{str(messages[-1].content).strip()}"
        )

    return None

@dynamic_prompt#每一次在生成提示词之前，调用此函数
def creation_plan_prompt_switch(request:ModelRequest):#动态切换提示词 
    """
    load_sys_prompts()是系统提示词
    load_report_prompts()是完整创作方案提示词
    """
    is_creation_plan=request.runtime.context.get("creation_plan",False)
    if is_creation_plan is True:#如果是完整创作方案场景，返回对应提示词内容
        return load_report_prompts()

    return load_sys_prompts()
    
