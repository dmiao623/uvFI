import yaml
from typing import Any, List

from config import PROJECT_INFO_PATH

def set_project_info(
    project_dir: str,
    model_name: str,
    pose_dir: str
):
    project_info = {
        "project_dir": project_dir,
        "model_name": model_name,
        "pose_dir": pose_dir
    }
    with PROJECT_INFO_PATH.open("w") as f:
        yaml.dump(project_info, f, sort_keys=False, default_flow_style=False)

def _read_project_info(*key_paths: str) -> List[Any]:
    with PROJECT_INFO_PATH.open("r") as f:
        data = yaml.safe_load(f)

    def _follow_path(mapping: dict, path: str) -> Any:
        cur = mapping
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                raise KeyError(f"Key '{path}' not found in YAML data")
        return cur

    return [_follow_path(data, kp) for kp in key_paths]


