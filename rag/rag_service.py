from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from model.factory import Chat_Model


"""
总结服务类：用户提问，搜索参考资料
将提问和参考资料提交给模型，让模型总结回复
"""

class RagSummarizeService(object):
    def __init__(self):
        self.vector_store=VectorStoreService()                                          #向量存储
        self.retriever=self.vector_store.get_retriever()                                #检索器
        self.prompts_text=load_rag_prompts()                                            #提示词文本
        self.prompts_template=PromptTemplate.from_template(self.prompts_text)           #提示词模版
        self.model=Chat_Model                                                           #model
        self.chain=self._init_chain()                                                 #链

    def _init_chain(self):
        chain=self.prompts_template | self.model | StrOutputParser()
        return chain

    def retriever_docs(self,query:str)->list[Document]:#检索文档函数
        return self.retriever.invoke(query)


    def rag_summarize(self,query:str)->str:
        context_docs=self.retriever_docs(query)#参考资料文档
        context=""
        counter=0
        for doc in context_docs:
            counter+=1
            context+=f"【参考资料{counter}】:参考资料{doc.page_content} | 参考元数据：{doc.metadata}\n"

        return self.chain.invoke(
            {
                "input":query,
                "context":context,
            }
        )

# if __name__=="__main__":
#     rag=RagSummarizeService()
#     print(rag.rag_summarize("美术师范专业有哪些就业方向"))