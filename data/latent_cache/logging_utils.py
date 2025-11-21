from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
from datetime import datetime
from rich.console import Console

_console = Console()

def log(msg: str) -> None:
    _console.log(msg)

def log_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"ts": datetime.utcnow().isoformat() + "Z", **record}) + "\n")
