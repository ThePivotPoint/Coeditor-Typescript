import os
import json
from pathlib import Path
from typing import Any, List, Dict, Sequence

def pretty_print_dict(d: dict, indent: int = 2):
    print(json.dumps(d, indent=indent, ensure_ascii=False))

def scalar_stats(values: Sequence[float]) -> dict:
    import numpy as np
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)) if len(arr) else 0.0,
        "std": float(np.std(arr)) if len(arr) else 0.0,
        "min": float(np.min(arr)) if len(arr) else 0.0,
        "max": float(np.max(arr)) if len(arr) else 0.0,
        "count": int(len(arr)),
    }

def join_list(lists: Sequence[Sequence[Any]]) -> List[Any]:
    out = []
    for l in lists:
        out.extend(l)
    return out

def rec_add_dict_to(target: dict, source: dict):
    for k, v in source.items():
        if k in target and isinstance(target[k], dict) and isinstance(v, dict):
            rec_add_dict_to(target[k], v)
        else:
            target[k] = v

class JSONCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
    def cached(self, rel_path: str, func, remake=False):
        path = self.cache_dir / rel_path
        if remake or not path.exists():
            value = func()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2, ensure_ascii=False)
            return value
        else:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f) 