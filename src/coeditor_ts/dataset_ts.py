"""
TypeScript 数据集处理模块
处理 TypeScript 项目的代码分析和问题生成
"""

import shutil
import tempfile
import traceback
import os
from pathlib import Path
from typing import Any, Sequence, Dict, List, TypedDict
from dataclasses import dataclass, field
from coeditor.common import *
from coeditor.common import repr_modified_args, WORK_DIR
from coeditor_ts.c3problem_ts import TSC3ProblemGenerator, TSC3ProblemTokenizer, TSC3ProblemChangeInlining
from coeditor_ts._utils import join_list
from coeditor_ts.ts_treesitter_utils import parse_ts_file_with_treesitter, diff_ast
import subprocess
import concurrent.futures
import time


@dataclass
class TSC3CombinedEncoder:
    change_processor: Any = field(default_factory=TSC3ProblemGenerator)
    problem_tranform: Any = field(default_factory=TSC3ProblemChangeInlining)
    tokenizer: Any = field(default_factory=TSC3ProblemTokenizer)

# --- JSON Cache (like PickleCache) ---
class JSONCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
    def cached(self, rel_path: str, func, remake=False):
        path = self.cache_dir / rel_path
        if remake or not path.exists():
            value = func()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                import json
                json.dump(value, f, indent=2, ensure_ascii=False)
            return value
        else:
            with open(path, "r", encoding="utf-8") as f:
                import json
                return json.load(f)

# --- Core dataset processing (TS version, stub logic) ---
def _process_commits_ts(root: Path, workdir: Path, is_training: bool, max_history_per_repo: int, change_processor: Any, cache: JSONCache, time_limit_per_commit: float = 10.0) -> Dict[str, Any]:
    # For TS, just load problems.json if exists, else empty list
    problems_file = root / "problems.json"
    if problems_file.exists():
        with open(problems_file, "r", encoding="utf-8") as f:
            return {"edits": json.load(f), "stats": {}}
    return {"edits": [], "stats": {}}

def get_commit_history_ts(repo_path: Path, max_history: int = 1000) -> List[str]:
    """获取仓库的 commit 哈希列表（从旧到新）"""
    result = subprocess.run([
        "git", "-C", str(repo_path), "rev-list", "--reverse", f"--max-count={max_history}", "HEAD"
    ], capture_output=True, text=True)
    return result.stdout.strip().splitlines()

def checkout_file_at_commit(repo_path: Path, file_path: Path, commit: str, out_path: Path) -> bool:
    """将指定 commit 的文件内容导出到 out_path"""
    rel_path = file_path.relative_to(repo_path)
    result = subprocess.run([
        "git", "-C", str(repo_path), "show", f"{commit}:{rel_path}"], capture_output=True)
    if result.returncode == 0:
        with open(out_path, "wb") as f:
            f.write(result.stdout)
        return True
    return False

def _process_single_project_ts(args):
    repo_path, change_processor, max_history_per_repo = args
    try:
        commits = get_commit_history_ts(repo_path, max_history=max_history_per_repo)
        ts_files = [p for p in repo_path.rglob("*.ts") if p.is_file() and "node_modules" not in str(p)]
        repo_problems = []
        for i in range(1, len(commits)):
            commit_old = commits[i-1]
            commit_new = commits[i]
            for ts_file in ts_files:
                tmp_old = Path("/tmp/ts_old.ts")
                tmp_new = Path("/tmp/ts_new.ts")
                ok_old = checkout_file_at_commit(repo_path, ts_file, commit_old, tmp_old)
                ok_new = checkout_file_at_commit(repo_path, ts_file, commit_new, tmp_new)
                if not (ok_old and ok_new):
                    continue
                tree_old, code_old = parse_ts_file_with_treesitter(str(tmp_old))
                tree_new, code_new = parse_ts_file_with_treesitter(str(tmp_new))
                changes = diff_ast(tree_old, code_old, tree_new, code_new)
                for change in changes:
                    problem = {
                        "repo": str(repo_path),
                        "file": str(ts_file),
                        "commit_old": commit_old,
                        "commit_new": commit_new,
                        "change_type": change.change_type,
                        "node_type": change.node.type,
                        "node_name": change.node.name,
                        "start_line": change.node.start_line,
                        "end_line": change.node.end_line,
                    }
                    repo_problems.append(problem)
        return (repo_path, repo_problems)
    except Exception as e:
        import traceback
        print(f"Error processing {repo_path}: {e}")
        traceback.print_exc()
        return (repo_path, [])

def dataset_from_projects_ts(
    cache: Any,
    project_roots: Sequence[Path],
    change_processor: Any,
    repo_training: Sequence[bool],
    max_history_per_repo: int,
    time_limit_per_commit: float,
    workers: int = 4,
) -> Dict[Path, List[Any]]:
    """
    并发处理每个项目，自动遍历 git commit 历史，做结构 diff，生成结构化变更问题对象。
    支持每项目缓存、异常处理和统计。
    """
    out = {}
    stats = {"total": 0, "success": 0, "fail": 0, "errors": []}
    args_list = [(repo_path, change_processor, max_history_per_repo) for repo_path in project_roots]
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for args in args_list:
            repo_path = args[0]
            cache_key = str(repo_path.resolve().name)
            # 先查缓存
            cached = None
            try:
                cached = cache.cached(cache_key, lambda: None)
            except Exception:
                cached = None
            if cached:
                out[repo_path] = cached
                stats["success"] += 1
            else:
                futures[executor.submit(_process_single_project_ts, args)] = (repo_path, cache_key)
        for future in concurrent.futures.as_completed(futures):
            repo_path, cache_key = futures[future]
            try:
                _, repo_problems = future.result()
                out[repo_path] = repo_problems
                cache.cached(cache_key, lambda: repo_problems, remake=True)
                stats["success"] += 1
            except Exception as e:
                import traceback
                print(f"[ERROR] Failed to process {repo_path}: {e}")
                traceback.print_exc()
                out[repo_path] = []
                stats["fail"] += 1
                stats["errors"].append(str(repo_path))
            stats["total"] += 1
    elapsed = time.time() - start_time
    print(f"[TS dataset] Projects processed: {stats['total']}, Success: {stats['success']}, Fail: {stats['fail']}, Time: {elapsed:.2f}s")
    if stats["fail"] > 0:
        print(f"[TS dataset] Failed projects: {stats['errors']}")
    return out

def datasets_from_repo_splits_ts(cache: JSONCache, repos_root: Path, change_processor: Any, splits: Sequence[str] = ("test", "valid", "train"), max_history_per_repo: int = 1000, time_limit_per_commit: float = 10.0, workers: int = 4) -> Dict[str, List[Any]]:
    projects = {split: [] for split in splits}
    split_is_training = {split: [] for split in splits}
    for split in splits:
        abs_path = Path(WORK_DIR, repos_root / split)
        if not abs_path.exists():
            continue
        ps = [p for p in abs_path.iterdir() if p.is_dir()]
        projects[split] = ps
        training = split == "train"
        split_is_training[split] = [training] * len(ps)
    # 保持和 Python 端一致，repo_training 为所有项目的训练标记
    all_projects = join_list(list(projects.values()))
    all_repo_training = join_list(list(split_is_training.values()))
    dataset = dataset_from_projects_ts(
        cache,
        all_projects,
        change_processor=change_processor,
        repo_training=all_repo_training,
        time_limit_per_commit=time_limit_per_commit,
        max_history_per_repo=max_history_per_repo,
        workers=workers,
    )
    return {k: join_list([dataset.get(r, []) for r in repos]) for k, repos in projects.items()}
# --- Dataset Types ---
class TSC3ProblemDataset(TypedDict):
    train: List[Any]
    valid: List[Any]
    test: List[Any]

# --- Main entry: make_or_load_ts_dataset ---
def make_or_load_ts_dataset(
    dataset_name: str,
    change_processor: Any,
    splits: Sequence[str],
    remake_problems: bool = False,
    time_limit_per_commit: float = 10.0,
    workers: int = 4,
) -> TSC3ProblemDataset:
    processed_dir = get_dataset_dir(dataset_name) / "processed"
    prob_config = repr_modified_args(change_processor)
    cache_dir = processed_dir / prob_config
    cache = JSONCache(cache_dir)
    if remake_problems:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
    results = datasets_from_repo_splits_ts(
        cache,
        get_dataset_dir(dataset_name) / "repos",
        change_processor,
        workers=workers,
        splits=splits,
        time_limit_per_commit=time_limit_per_commit,
    )
    size_mb = 0.0
    n = 0
    if cache_dir.exists():
        for f in cache_dir.iterdir():
            n += 1
            size_mb += f.stat().st_size / (1024**2)
    print(f"Dataset total size ({n=}): {size_mb:.2f} MB")
    return TSC3ProblemDataset(
        train=results.get("train", []),
        valid=results.get("valid", []),
        test=results.get("test", []),
    )

# --- Transformed dataset (stub logic) ---
def make_or_load_transformed_ts_dataset(
    dataset_name: str,
    dataset: TSC3ProblemDataset,
    encoder: TSC3CombinedEncoder,
    remake_problems: bool = False,
    workers: int = 4,
) -> Dict[str, List[Any]]:
    def transform_eval_problems(dataset: TSC3ProblemDataset) -> Dict[str, List[Any]]:
        results = {}
        for split in ("valid", "test", "train"):
            prob_list = dataset.get(split, [])
            results[split] = encoder.problem_tranform.transform(prob_list)
        return results
    from coeditor.common import get_dataset_dir
    import json
    transformed_dir = get_dataset_dir(dataset_name) / "transformed"
    cache = JSONCache(transformed_dir)
    return cache.cached(
        f"transformed.json",
        lambda: transform_eval_problems(dataset),
        remake=remake_problems,
    )

# --- Save/load helpers ---
def save_datasets_ts(datasets: Dict[str, Any], save_dir: Path) -> None:
    import json
    for name, dataset in datasets.items():
        with open(save_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
    os.system(f"du -sh {save_dir}")

def load_datasets_ts(save_dir: Path, splits=("test", "valid", "train")) -> Dict[str, Any]:
    import json
    return {
        name: json.load(open(save_dir / f"{name}.json", "r", encoding="utf-8"))
        for name in splits
        if (save_dir / f"{name}.json").exists()
    } 