from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

def load_sys_prompts() -> str:
    try:
        system_prompt_path= get_abs_path(prompts_conf["main_prompts_path"])
    except KeyError as e:
        logger.error(f"[load_sys_prompts]can't find main_prompts_path")
        raise e


    try:
        return open(system_prompt_path,"r",encoding="utf_8").read()
    except Exception as e:
        logger.error(f"[load_system_prompts]error.{str(e)}")
        raise e


def load_rag_prompts() -> str:
    try:
        rag_prompt_path= get_abs_path(prompts_conf["rag_summarize_prompts_path"])
    except KeyError as e:
        logger.error(f"[load_RAG_prompts]can't find rag_summarize_prompts_path")
        raise e


    try:
        return open(rag_prompt_path,"r",encoding="utf_8").read()
    except Exception as e:
        logger.error(f"[load_RAG_prompts]error.{str(e)}")
        raise e

def load_report_prompts() -> str:
    try:
        report_prompt_path= get_abs_path(prompts_conf["report_prompts_path"])
    except KeyError as e:
        logger.error(f"[load_report_prompts]can't find report_prompts_path")
        raise e


    try:
        return open(report_prompt_path,"r",encoding="utf_8").read()
    except Exception as e:
        logger.error(f"[load_report_prompts]error.{str(e)}")
        raise e

if __name__=='__main__':
    print(load_sys_prompts())
    print(load_rag_prompts())
    print(load_report_prompts())
