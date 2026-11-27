from pathlib import Path

def get_project_root():
    # Nel container viene montato il repository intero; il progetto PageRank
    # vive quindi nella sottocartella pre-processing.
    docker_root = Path("/opt/spark/work-dir/project")
    docker_pagerank_root = docker_root / "00_simulate/pre-processing/pagerank"
    if docker_pagerank_root.exists():
        return docker_pagerank_root
    return Path(__file__).resolve().parents[1]
