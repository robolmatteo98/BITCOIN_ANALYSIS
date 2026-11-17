from glob import glob
from pathlib import Path

def select_dat_files_from_glob(glob_pattern, sample_files=None):
    files = sorted(Path(path) for path in glob(glob_pattern))
    if sample_files:
        return files[:sample_files]
    return files