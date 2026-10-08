from os import PathLike
from pathlib import Path

#路径处理工具

def get_abs_path(path: str | PathLike[str]) -> str:#获取路径的规范化绝对路径
    
    input_path = Path(path)
    project_root = Path(__file__).resolve().parent.parent
    return str(input_path.resolve() if input_path.is_absolute() else (project_root / input_path).resolve())

# if __name__ == "__main__":
#     print(get_abs_path("config/config.txt"))
