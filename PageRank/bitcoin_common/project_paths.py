from pathlib import Path

def get_project_root():
    docker_root = Path("/opt/spark/work-dir/project")
    if docker_root.exists():
        return docker_root
    return Path(__file__).resolve().parents[1]