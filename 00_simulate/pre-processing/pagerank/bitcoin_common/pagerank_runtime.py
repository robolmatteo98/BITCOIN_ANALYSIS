import tempfile
import zipfile

from bitcoin_common.project_paths import get_project_root

# configurazione per docker
# serve a spedire ai worker Spark i file Python del progetto, definendo le cartelle da impacchettare, creano un file .zip e Saprk si occupa di mandare i file .zip ai worker.
def add_python_files_to_workers(sc):
    project_root = get_project_root()
    src_dir = project_root / "bitcoin_pagerank" / "src"
    utils_dir = src_dir / "utils"
    common_dir = project_root / "bitcoin_common"

    for source_dir, arc_base in (
        (utils_dir, src_dir),
        (common_dir, project_root),
    ):
        with tempfile.NamedTemporaryFile(prefix="bitcoin_pagerank_", suffix=".zip", delete=False) as tmp_zip:
            zip_path = tmp_zip.name
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in sorted(source_dir.rglob("*.py")):
                archive.write(file_path, arcname=file_path.relative_to(arc_base))
        sc.addPyFile(zip_path)