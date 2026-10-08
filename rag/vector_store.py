from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
from model.factory import Embeding_Model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader,txt_loader,list_dir_with_allowed,get_file_md5_hex
import os
from utils.logger_handler import logger
from langchain_core.documents import Document

#向量存储服务
class VectorStoreService:
    def __init__(self):
        self.vector_store=Chroma(#初始化一个向量服务器
            collection_name=chroma_conf["collection_name"],
            embedding_function=Embeding_Model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"])

        )
        self.spliter=RecursiveCharacterTextSplitter(#初始化一个文本分割器
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):#获取检索器对象(检索k段相关文本)
        return self.vector_store.as_retriever(search_kwargs={"k":chroma_conf["k"]})


    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量，存入向量库
        计算文件的md5做去重
        """
        def check_md5_hex(md5_for_check:str):#md5是否处理过（去重,用于文本内容存入向量库之前

            md5_store_path=get_abs_path(chroma_conf["md5_hex_store"])
            if not os.path.exists(md5_store_path):#判断存md5的文件是否存在
                #如果文件不存在则创建文件（打开后关闭)
                open(md5_store_path,"w",encoding="utf-8").close()
                return False

            with open(md5_store_path,"r",encoding="utf-8") as f:
                for line in f.readlines():
                    line=line.strip()
                    if line == md5_for_check:
                        return True
                return False


        def save_md5_hex(md5_for_check:str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]),"a",encoding="utf-8") as f:
                f.write(md5_for_check+"\n")


        def get_file_documents(read_path):#pdf和txt保存为Document对象
            if read_path.lower().endswith(".txt"):
                return txt_loader(read_path)
            if read_path.lower().endswith(".pdf"):
                return pdf_loader(read_path)

            return []

        allowed_file_path=list_dir_with_allowed(#待处理的文件路径
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"])
        )

        for path in allowed_file_path:
            #获取文件的md5
            md5_hex=get_file_md5_hex(path)
            if md5_hex is None:
                logger.warning(f"[加载知识库]{path}无法计算MD5, 跳过")
                continue

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已存在知识库，跳过")
                continue

            try:
                documents:list[Document]=get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文件，跳过")
                    continue

                split_document=self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                #将内容存入向量库
                self.vector_store.add_documents(split_document)

                #记录md5，避免下次重复传入
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path}内容加载成功")
            except Exception as e:
                #exc_info=True会记录详细报错堆栈
                logger.error(f"加载知识库{path}失败：{str(e)}",exc_info=True)


if __name__=="__main__":
    vs=VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()
    res=retriever.invoke("美术师范专业有哪些就业方向")
    for r in res:
        print(r.page_content)
        print("-"*20)
