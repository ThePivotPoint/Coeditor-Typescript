from typing import Sequence, TypeVar, Generic, Iterable, Mapping, Any
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import pickle
import warnings
from coeditor._utils import pretty_print_dict, scalar_stats
from coeditor.common import get_dataset_dir, repr_modified_args, join_list, not_none
from coeditor.dataset import PickleCache, pmap
from coeditor_ts.c3problem import TsC3ProblemGenerator, TsC3Problem, TsC3ProblemTokenizer
from prepare_data_ts import TsC3ProblemChangeInlining
from coeditor_ts.common import *
from coeditor.git import get_commit_history
import shutil
import os
import gc
import time
from pathlib import Path
from typing import Sequence
from coeditor_ts.common import get_typescript_files
from coeditor_ts.scoped_changes import parse_typescript_module_script, TsModuleChange
from coeditor.change import Added, Deleted, Modified
from coeditor.git import CommitInfo
import copy

TProb = TypeVar("TProb", bound=TsC3Problem)

@dataclass
class TsTokenizedEditDataset(Generic[TProb]):
    _edits: list[TProb]

    def __repr__(self) -> str:
        n_edits = len(self.all_edits())
        return f"TsTokenizedEditDataset(n_edits={n_edits})"

    def subset_edits(self, n_edits: int) -> "TsTokenizedEditDataset":
        return TsTokenizedEditDataset.from_edits(self.all_edits()[:n_edits])

    def overall_stats(self) -> dict:
        all_edits = self.all_edits()
        # 这里假设 change_type/Added 逻辑已在 TS 迁移
        n_added = sum(getattr(e, 'change_type', None) == 'Added' for e in all_edits)
        basic_stats = {
            "n_edits": len(all_edits),
            "n_additions": n_added,
        }
        # TypeScript 版暂不统计 stats 字段，后续如有需要可补充
        return basic_stats

    def all_edits(self) -> list[TProb]:
        return self._edits

    @staticmethod
    def from_edits(edits: Iterable[TProb]) -> "TsTokenizedEditDataset[TProb]":
        return TsTokenizedEditDataset(list(edits))

@dataclass
class TsC3CombinedEncoder:
    change_processor: TsC3ProblemGenerator = field(default_factory=TsC3ProblemGenerator)
    problem_tranform: TsC3ProblemChangeInlining = field(default_factory=TsC3ProblemChangeInlining)
    edit_tokenizer: TsC3ProblemTokenizer = field(default_factory=TsC3ProblemTokenizer)

@dataclass
class _ProcessingResult:
    edits: Sequence
    stats: dict[str, dict | Any]


def edits_from_ts_commit_history(
    project_dir: Path,
    history: Sequence[CommitInfo],
    tempdir: Path,
    change_processor,
    silent: bool = False,
    time_limit: float = -1.0,  # Use -1.0 to indicate no time limit
):
    """
    从 TypeScript 仓库提交历史生成编辑问题对象（scope_change 版本）
    """
    tempdir = tempdir.resolve()
    if tempdir.exists():
        raise FileExistsError(f"Workdir '{tempdir}' already exists.")
    tempdir.mkdir(parents=True, exist_ok=False)
    results = []
    start_time = time.time()

    def has_timeouted(step):
        if time_limit < 0:
            return False
        return (time.time() - start_time > time_limit)

    try:
        # 1. 复制 .git 目录
        shutil.copytree(project_dir / ".git", tempdir / ".git")

        # 2. 获取所有 ts/tsx 文件
        def get_modules(proj_dir):
            modules = {}
            for f in get_typescript_files(proj_dir):
                abs_path = proj_dir / f
                if abs_path.exists():
                    mod, _ = parse_typescript_module_script(proj_dir, abs_path)
                    modules[f] = mod
            return modules

        # 3. checkout 到最新 commit
        import subprocess
        def checkout_commit(commit_hash):
            subprocess.run(
                ["git", "checkout", "-f", commit_hash],
                cwd=tempdir,
                capture_output=True,
                check=True,
            )

        # 4. 初始化模块状态
        commit_now = history[-1]
        checkout_commit(commit_now.hash)
        path2module = get_modules(tempdir)

        # 5. 反向遍历 commit 历史
        future_commits = list(reversed(history[:-1]))
        for step, commit_next in enumerate(future_commits):
            if has_timeouted(step):
                return results

            # 获取变更文件
            changed_files = subprocess.run(
                [
                    "git", "diff", "--no-renames", "--name-status",
                    commit_now.hash, commit_next.hash
                ],
                cwd=tempdir,
                capture_output=True, text=True
            ).stdout.splitlines()

            path_changes = set()
            for line in changed_files:
                segs = line.split("\t")
                if len(segs) == 2:
                    tag, path = segs
                    if not (path.endswith(".ts") or path.endswith(".tsx")):
                        continue
                    if tag.endswith("A"):
                        path_changes.add(Added(path))
                    elif tag.endswith("D"):
                        path_changes.add(Deleted(path))
                    if tag.endswith("M"):
                        path_changes.add(Modified(path, path))
                elif len(segs) == 3:
                    tag, path1, path2 = segs
                    assert tag.startswith("R")
                    if path1.endswith(".ts") or path1.endswith(".tsx"):
                        path_changes.add(Deleted(path1))
                    if path2.endswith(".ts") or path2.endswith(".tsx"):
                        path_changes.add(Added(path2))

            # 深拷贝将要变更的模块
            to_copy = {c.before for c in path_changes if hasattr(c, 'before')}
            for k in to_copy:
                if k in path2module:
                    path2module[k] = copy.deepcopy(path2module[k])

            # 切换到下一个 commit
            checkout_commit(commit_next.hash)
            new_path2module = get_modules(tempdir)

            # 生成模块变更
            changed = {}
            for path_change in path_changes:
                # 兼容 earlier/after 可能为 None 的情况
                path_str = getattr(path_change, 'earlier', None)
                if path_str is None:
                    path_str = getattr(path_change, 'after', None)
                if path_str is None:
                    continue  # 跳过无效 path_change
                path = tempdir / path_str
                rel_path = path.relative_to(tempdir)
                if isinstance(path_change, Added):
                    if rel_path.exists():
                        mod, _ = parse_typescript_module_script(tempdir, rel_path)
                        new_path2module[rel_path] = mod
                        changed[mod.mname] = TsModuleChange.from_modules(Added(mod))
                elif isinstance(path_change, Deleted):
                    if rel_path in new_path2module:
                        mod = new_path2module.pop(rel_path)
                        changed[mod.mname] = TsModuleChange.from_modules(Deleted(mod))
                elif isinstance(path_change, Modified):
                    if rel_path in new_path2module:
                        mod_new, _ = parse_typescript_module_script(tempdir, rel_path)
                        mod_old = path2module[rel_path]
                        new_path2module[rel_path] = mod_new
                        changed[mod_new.mname] = TsModuleChange.from_modules(Modified(mod_old, mod_new))

            # 生成问题对象
            processed = change_processor.process_change(
                changed,  # 传递模块变更
                # 可根据需要传递 pre/post 分析等
            )
            results.extend(processed)
            commit_now = commit_next
            path2module = new_path2module

    finally:
        shutil.rmtree(tempdir)
        gc.collect()

    return results

def _process_ts_commits(root, workdir, is_training, max_history_per_repo, change_processor, cache, time_limit_per_commit=10.0):
    key = f"{root.name}({max_history_per_repo}, {is_training=})"
    commits = []
    if not cache.contains(key):
        commits = get_commit_history(root)[-max_history_per_repo:]
    try:
        edits = cache.cached(
            key,
            lambda: edits_from_ts_commit_history(
                root,
                commits,
                tempdir=workdir / "code" / root.name,
                change_processor=change_processor,
                silent=True,
                time_limit=time_limit_per_commit * (len(commits) + 10),
            ),
        )
    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        warnings.warn(f"Failed to process project: {root}\nError: {e}")
        import traceback
        traceback.print_exception(e, limit=-6)
        edits = []
    stats = dict()
    return _ProcessingResult(edits=edits, stats=stats)

def ts_dataset_from_projects(
    cache: PickleCache,
    project_roots: Sequence[Path],
    change_processor: TsC3ProblemGenerator,
    repo_training: Sequence[bool],
    max_history_per_repo: int,
    time_limit_per_commit: float,
    workers: int = 8,
) -> Mapping[Path, Sequence[TProb]]:
    """
    并行处理多个 TypeScript 仓库，提取编辑问题对象。
    """
    import os
    import tempfile
    from pathlib import Path

    pid = os.getpid()
    workdir = Path(tempfile.gettempdir()) / "ts_dataset_from_projects" / f"pid-{pid}"

    roots = project_roots
    workdirs = [workdir / f"repo-{i}" for i in range(len(roots))]

    try:
        presults = pmap(
            _process_ts_commits,
            roots,
            workdirs,
            repo_training,
            key_args={
                "max_history_per_repo": max_history_per_repo,
                "change_processor": change_processor,
                "time_limit_per_commit": time_limit_per_commit,
                "cache": cache,
            },
            max_workers=workers,
            tqdm_args={"unit": "repo"},
        )
    finally:
        if workdir.exists():
            print("Removing workdir:", workdir)
            import shutil
            shutil.rmtree(workdir)

    project2edits = dict()
    for root, pr in zip(roots, presults):
        project2edits.setdefault(root, []).extend(pr.edits)
    return project2edits

def ts_datasets_from_repo_splits(
    cache: PickleCache,  # 【guohx】缓存对象，用于加速重复处理
    repos_root: Path,  # 【guohx】原始仓库根目录路径
    change_processor: TsC3ProblemGenerator,  # 【guohx】变更处理器，负责从Git变更生成问题对象
    splits: Sequence[str] = ("test", "valid", "train"),  # 【guohx】要处理的数据分集，默认处理所有分集
    max_history_per_repo: int = 1000,  # 【guohx】每个仓库最多处理的commit数量，避免处理过长的历史
    time_limit_per_commit: float = 10.0,  # 【guohx】每个commit的处理时间限制（秒）
    workers: int = DefaultWorkers,  # 【guohx】并行处理的worker数量
) -> dict[str, Sequence[TProb]]:  # 【guohx】返回按分集组织的问题字典，键为分集名，值为问题列表
    # 【guohx】初始化项目路径和训练标志的存储字典
    projects = dict[str, list[Path]]()  # 【guohx】存储每个分集下的项目路径列表
    split_is_training = dict[str, list[bool]]()  # 【guohx】存储每个分集是否为训练集的标志列表
    
    # 【guohx】获取工作目录的绝对路径
    abs_path = os.path.join(WORK_DIR,)
    
    # 【guohx】遍历每个分集，收集该分集下的所有仓库项目
    for split in splits:
        # 【guohx】构建分集的完整路径：WORK_DIR/repos_root/split
        abs_path = Path(WORK_DIR, repos_root / split)
        
        # 【guohx】检查分集目录是否存在，不存在则跳过并警告
        if not (abs_path).exists():
            warnings.warn(f"Split {split} not found at {abs_path}.")
            continue
        
        # 【guohx】收集该分集目录下的所有子目录（每个子目录对应一个仓库项目）
        ps = [p for p in (abs_path).iterdir() if p.is_dir]
        projects[split] = ps  # 【guohx】将项目路径列表存储到projects字典中
        
        # 【guohx】设置训练标志：只有train分集为True，其他分集为False
        training = split == "train"
        split_is_training[split] = [training] * len(ps)  # 【guohx】为每个项目设置相同的训练标志
        
        # 【guohx】如果分集下没有找到任何项目，输出警告
        if not ps:
            warnings.warn(f"No projects found in {split} split")

    # 【guohx】调用dataset_from_projects进行批量处理
    # 【guohx】将所有分集的项目路径合并，统一处理，提高效率
    dataset = ts_dataset_from_projects(
        cache,  # 【guohx】缓存对象
        join_list(projects.values()),  # 【guohx】合并所有分集的项目路径列表
        change_processor=change_processor,  # 【guohx】变更处理器
        repo_training=join_list(split_is_training.values()),  # 【guohx】合并所有分集的训练标志列表
        time_limit_per_commit=time_limit_per_commit,  # 【guohx】commit处理时间限制
        max_history_per_repo=max_history_per_repo,  # 【guohx】每个仓库的最大历史长度
        workers=workers,  # 【guohx】并行worker数
    )
    
    # 【guohx】重新按分集组织结果，将dataset_from_projects返回的项目级别结果转换为分集级别
    # 【guohx】对每个分集，收集该分集下所有仓库生成的问题，合并为一个列表
    return {k: join_list(dataset[r] for r in repos) for k, repos in projects.items()}


class TsC3ProblemDataset(dict):
    def __init__(self, **splits):
        super().__init__(splits)
        for k, v in splits.items():
            setattr(self, k, v)

def make_or_load_ts_dataset(
    dataset_name: str,
    change_processor: TsC3ProblemGenerator,
    splits: Sequence[str],
    remake_problems: bool = False,
    time_limit_per_commit: float = 10.0,
    workers: int = 8,
) -> TsC3ProblemDataset:
    """
    生成或加载 TypeScript 结构化编辑问题数据集。
    """
    from coeditor.common import get_dataset_dir, repr_modified_args
    prob_config = repr_modified_args(change_processor)
    processed_dir = get_dataset_dir(dataset_name) / "processed"
    cache_dir = processed_dir / prob_config
    cache = PickleCache(cache_dir)
    if remake_problems:
        cache.clear()
    results = ts_datasets_from_repo_splits(
        cache,
        get_dataset_dir(dataset_name) / "repos",
        change_processor,
        workers=workers,
        splits=splits,
        time_limit_per_commit=time_limit_per_commit,
    )
    return TsC3ProblemDataset(
        train=results.get("train", []),
        valid=results.get("valid", []),
        test=results.get("test", []),
    )

def make_or_load_ts_transformed_dataset(
    dataset_name: str,
    dataset: TsC3ProblemDataset | None,
    encoder: TsC3CombinedEncoder,
    remake_problems: bool = False,
    workers: int = 8,
) -> dict[str, Sequence[TsC3Problem]]:
    def transform_eval_problems(dataset: TsC3ProblemDataset) -> dict[str, Sequence[TsC3Problem]]:
        results = dict[str, Sequence[TsC3Problem]]()
        for split in ("valid", "test"):
            prob_lists = pmap(
                encoder.problem_tranform.transform,
                dataset[split],
                desc=f"transform({split})",
                chunksize=1000,
                max_workers=workers,
            )
            results[split] = join_list(prob_lists)
        return results

    proc_config = repr_modified_args(encoder.change_processor)
    trans_config = repr_modified_args(encoder.problem_tranform)
    transformed_dir = get_dataset_dir(dataset_name) / "transformed"
    cache = PickleCache(transformed_dir)
    return cache.cached(
        f"eval-{proc_config}-{trans_config}",
        lambda: transform_eval_problems(not_none(dataset)),
        remake=remake_problems,
    )

def save_ts_datasets(datasets: Mapping[str, Any], save_dir: Path) -> None:
    for name, dataset in datasets.items():
        with open(save_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(dataset, f)
    subprocess.run(["du", "-sh", str(save_dir)])

def load_ts_datasets(save_dir: Path, splits=("test", "valid", "train")) -> dict[str, Any]:
    return {
        name: pickle.load(open(save_dir / f"{name}.pkl", "rb"))
        for name in splits
        if (save_dir / f"{name}.pkl").exists()
    }

def get_ts_repo_signature(repo: Path, n_commits: int = 30) -> tuple[str, ...]:
    # TODO: 迁移 TypeScript 版本的 get_commit_history
    raise NotImplementedError("get_ts_repo_signature 需实现 TypeScript 仓库签名逻辑") 