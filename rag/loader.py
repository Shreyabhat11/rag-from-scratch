from pathlib import Path
from typing import List


def load_documents(data_dir: str = "data") -> List[str]:
    """Read all .txt files in data_dir, return list of strings."""
    docs = []
    for path in Path(data_dir).glob("*.txt"):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append(text)
    return docs