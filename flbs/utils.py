import os

def get_git_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def relative_to_absolute_path(path: str) -> str:
    return os.path.abspath(os.path.join(get_git_root(), path))