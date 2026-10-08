import os
import hashlib
from pathlib import Path
from utils.logger_handler import logger 
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

#文件处理工具


def get_file_md5_hex(file_path: str):#获取文件的md5的十六进制字符串，用于RAG去重

    if not os.path.exists(file_path):#文件不存在
        logger.error(f"File not found: {file_path}")
        return None
    if not os.path.isfile(file_path):#非文件（文件夹）
        logger.error(f"File is not a file: {file_path}")
        return None

    md5_hash = hashlib.md5()

    #流式分片计算md5，避免大文件占用过多内存
    chunk_size = 4096  # 4KB分片
    try:
        with open(file_path, "rb") as f:    #必须二进制读取
            chunk=f.read(chunk_size)
            while chunk:
                md5_hash.update(chunk)
                chunk = f.read(chunk_size)
            md5_hex=md5_hash.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"Error calculating MD5 for file {file_path}: {e}")
        return None



def list_dir_with_allowed(path:str,allowed_types:tuple[str,...]):#返回文件夹中，允许访问类型的文件列表
    files=[]

    if not os.path.isdir(path):
        logger.error(f"Path is not a directory: {path}")
        return ()
    
    normalized_types=tuple(item.lower() for item in allowed_types)
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.lower().endswith(normalized_types):
                files.append(os.path.join(root,filename))
    return tuple(sorted(files))


def pdf_loader(filepath: str)->list[Document]:#加载pdf文件，返回文本内容
    documents=PyPDFLoader(filepath).load()
    category=Path(filepath).parent.name
    for document in documents:
        document.metadata.update({
            "source_file": Path(filepath).name,
            "category": category,
        })
    return documents
    


def txt_loader(filepath: str)->list[Document]:#加载txt文件，返回文本内容
    """读取 UTF-8 文本，并解析可选的 YAML 风格头部元数据。

    文件开头可使用如下格式：
    ---
    title: 计算机科学与技术
    major: 计算机科学与技术
    source_scope: zstu_official
    ---
    正文……

    这里只解析简单的 ``键: 值``，避免额外引入配置依赖或执行任意 YAML。
    """
    with open(filepath,"r",encoding="utf-8-sig") as file:
        content=file.read()

    metadata={
        "source": str(Path(filepath).resolve()),
        "source_file": Path(filepath).name,
        "category": Path(filepath).parent.name,
    }

    lines=content.splitlines()
    if lines and lines[0].strip() == "---":
        closing_index=None
        for index,line in enumerate(lines[1:],start=1):
            if line.strip() == "---":
                closing_index=index
                break

        if closing_index is not None:
            for line in lines[1:closing_index]:
                if ":" not in line:
                    continue
                key,value=line.split(":",1)
                key=key.strip()
                value=value.strip()
                if key:
                    metadata[key]=value
            content="\n".join(lines[closing_index+1:]).strip()

    if not content:
        return []
    return [Document(page_content=content,metadata=metadata)]

