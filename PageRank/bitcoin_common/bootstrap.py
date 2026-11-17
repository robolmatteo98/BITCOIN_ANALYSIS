import sys
from pathlib import Path

def configure_imports(reference_file):
    ref_path = Path(reference_file).resolve()
    src_dir = ref_path.parent
    project_root = src_dir.parent.parent

    for path in (src_dir, project_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)