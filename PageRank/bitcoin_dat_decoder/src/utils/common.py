import shutil
from pathlib import Path

def clean_output_dir(path):
    output = Path(path)
    shutil.rmtree(output) # cancella tutto il contenuto della cartella

# funzione che mi convalida il path
def ensure_dir(path):
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output