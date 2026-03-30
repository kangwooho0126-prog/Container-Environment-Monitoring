import re
from pathlib import Path


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def norm_col(s):
    s = str(s).strip().lower().replace("\u3000", " ")
    return re.sub(r"\s+", "_", s)