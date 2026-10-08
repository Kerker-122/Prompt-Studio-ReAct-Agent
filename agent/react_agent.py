from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from model.factory import Chat_Model
from utils.prompt_loader import load_sys_prompts
from agent.tools.agent_tools import PROMPT_TOOLS
from agent.tools.middleware import monitor_tool,log_before_model,creation_plan_prompt_switch

class ReactAgent:
    def __init__(self):
        self.memory=InMemorySaver()#创建内存存储器，多轮对话存入内存机制
        self.agent=create_agent(
            model=Chat_Model,
            system_prompt=load_sys_prompts(),
            tools=PROMPT_TOOLS,
            middleware=[monitor_tool,log_before_model,creation_plan_prompt_switch],
            checkpointer=self.memory,#将内存存储器交给agent
        )

    def execute_stream(self,query:str,thread_id:str="default"):
        input_dict={ 
            "messages":[
                {"role":"user","content":query}
            ]
        }

        config={
            "configurable":{
                "thread_id":thread_id
            }
        }

        history_state=self.agent.get_state(config)
        seen_message_ids={
            message.id
            for message in history_state.values.get("messages",[])
            if message.id is not None
        }

        #context就是上下文runtime中的信息，就是提示词切换的标记
        for chunk in self.agent.stream(
            input_dict,
            config=config,
            stream_mode="values",
            context={"creation_plan":False}#默认全局会话模式
        ):
            latest_message=chunk["messages"][-1]
            content=latest_message.content
            message_id=latest_message.id
            if (
                isinstance(latest_message,AIMessage)
                and not latest_message.tool_calls
                and content
                and message_id not in seen_message_ids
            ):
                if message_id is not None:
                    seen_message_ids.add(message_id)
                yield str(content).strip()+"\n"
                
# if __name__=='__main__':
#     agent=ReactAgent()
#     for trunk in agent.execute_stream("帮我设计一个失忆剑客"):
#         print(trunk,end="",flush=True)


        
