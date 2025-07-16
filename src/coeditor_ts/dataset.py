from re import T
import shutil
import tempfile
import traceback
import os
from coeditor import scoped_changes
from coeditor._utils import pretty_print_dict, scalar_stats

from .c3problem import (
    TsC3Problem,
    TsC3ProblemGenerator,
    C3ProblemSimpleSplit,
    TsC3ProblemTokenizer,
    TsC3ProblemTransform,
    JediUsageAnalyzer,
    fix_ts_cache,
)
from .change import Added
from .common import *
from .encoding import TEdit
from .git import CommitInfo, get_commit_history
from .scoped_changes import ProjectChangeProcessor, TProb, edits_from_ts_commit_history
from pathlib import Path
from .common import WORK_DIR

@dataclass
class TokenizedEditDataset(Generic[TEdit]):
    _edits: list[TEdit]

    def __repr__(self) -> str:
        n_edits = len(self.all_edits())
        return f"TokenizedEditDataset(n_edits={n_edits})"

    def subset_edits(self, n_edits: int) -> "TokenizedEditDataset":
        return TokenizedEditDataset.from_edits(self.all_edits()[:n_edits])

    def overall_stats(self) -> dict:
        all_edits = self.all_edits()
        n_added = sum(isinstance(e.change_type, Added) for e in all_edits)
        basic_stats = {
            "n_edits": len(all_edits),
            "n_additions": n_added,
        }
        extra_stats = dict[str, list]()
        for e in all_edits:
            for k, v in e.stats().items():
                if k in extra_stats:
                    extra_stats[k].append(v)
                else:
                    extra_stats[k] = [v]
        return basic_stats | {k: scalar_stats(v) for k, v in extra_stats.items()}

    def all_edits(self) -> list[TEdit]:
        return self._edits

    @staticmethod
    def from_edits(edits: Iterable[TEdit]) -> "TokenizedEditDataset[TEdit]":
        return TokenizedEditDataset(list(edits))


@dataclass
class TsC3CombinedEncoder:
    change_processor: ProjectChangeProcessor[TsC3Problem] = field(
        default_factory=TsC3ProblemGenerator
    )
    problem_tranform: TsC3ProblemTransform = field(default_factory=TsC3ProblemTransform)
    edit_tokenizer: TsC3ProblemTokenizer = field(default_factory=TsC3ProblemTokenizer)


@dataclass
class _ProcessingResult:
    edits: Sequence
    stats: dict[str, dict | Any]


# 【guohx】处理单个代码仓库的提交历史，提取编辑信息并生成问题对象的内部函数
def _process_commits(
    root: Path,  # 【guohx】代码仓库的根目录路径
    workdir: Path,  # 【guohx】工作目录路径，用于存储临时文件
    is_training: bool,  # 【guohx】是否为训练模式，影响问题生成策略
    max_history_per_repo: int,  # 【guohx】每个仓库最多处理的commit数量
    change_processor: ProjectChangeProcessor[TsC3Problem],  # 【guohx】变更处理器，负责生成C3Problem对象
    cache: PickleCache,  # 【guohx】缓存对象，用于加速重复处理
    time_limit_per_commit: float = 10.0,  # 【guohx】每个commit的处理时间限制（秒），默认10秒
) -> _ProcessingResult:  # 【guohx】返回包含编辑列表和统计信息的处理结果
    # 【guohx】修复TypeScript缓存，确保使用进程特定的tree-sitter缓存目录
    fix_ts_cache(workdir)
    # 【guohx】清空时间日志记录器，准备记录本次处理的时间统计
    scoped_changes._tlogger.clear()
    # 【guohx】清空变更处理器的统计信息，准备收集新的统计
    change_processor.clear_stats()
    # 【guohx】设置变更处理器的训练模式标志
    change_processor.set_training(is_training)
    # 【guohx】生成缓存键，包含仓库名称、最大历史长度和训练模式信息
    key = f"{root.name}({max_history_per_repo}, {is_training=})"
    # 【guohx】初始化提交列表
    commits = []
    # 【guohx】检查缓存中是否已存在该仓库的处理结果
    if not cache.contains(key):
        # 【guohx】如果缓存中不存在，获取提交历史，保留最老的commits（限制数量）
        commits = get_commit_history(root)[-max_history_per_repo:]
        
        # 【guohx】记录提交信息到日志文件
        import logging
        logging.basicConfig(
            filename=f"commits_log_{root.name}.txt",
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        logging.info(f"Processing repository: {root.name}")
        logging.info(f"Total commits to process: {len(commits)}")
        for i, commit in enumerate(commits):
            logging.info(f"Commit {i+1}: {commit.hash} - {commit.msg[:100]}...")
        logging.info("-" * 80)
    try:
        # 【guohx】不能在这里直接返回，因为子进程可能在返回后被杀死
        # 【guohx】使用缓存机制获取或生成编辑信息
        
        edits = cache.cached(
            key,  # 【guohx】缓存键
            lambda: edits_from_ts_commit_history(  # 【guohx】如果缓存不存在，执行这个函数生成编辑
                root,  # 【guohx】仓库根目录
                commits,  # 【guohx】要处理的提交列表（TypeScript版本）
                tempdir=workdir / "code" / root.name,  # 【guohx】临时目录：workdir/code/仓库名
                change_processor=change_processor,  # 【guohx】变更处理器
                silent=True,  # 【guohx】静默模式，不输出详细信息
                time_limit=time_limit_per_commit * (len(commits) + 10),  # 【guohx】总时间限制：每个commit的时间限制 * (提交数+10)
            ),
        )
    except Exception as e:
        # 【guohx】如果是键盘中断异常，直接抛出
        if isinstance(e, KeyboardInterrupt):
            raise
        # 【guohx】如果是其他异常，输出警告信息并打印异常堆栈
        warnings.warn(f"Failed to process project: {root}\nError: {e}")
        traceback.print_exception(e, limit=-6)  # 【guohx】打印异常堆栈，限制显示最后6行
        # 【guohx】处理失败时，返回空的编辑列表
        edits = []
    # 【guohx】初始化统计信息字典
    stats = dict()
    # 【guohx】将变更处理器的统计信息添加到总统计中
    change_processor.append_stats(stats)
    # 【guohx】将时间日志记录器的统计信息添加到总统计中
    rec_add_dict_to(stats, {"tlogger": scoped_changes._tlogger.times})
    
    # 【guohx】记录处理结果到日志文件
    try:
        import logging
        logging.basicConfig(
            filename=f"commits_log_{root.name}.txt",
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        logging.info(f"Processing completed for repository: {root.name}")
        logging.info(f"Generated edits: {len(edits)}")
        logging.info(f"Stats: {stats}")
        logging.info("=" * 80)
    except Exception as e:
        print(f"Failed to write completion log: {e}")
    
    # 【guohx】返回包含编辑列表和统计信息的处理结果对象
    return _ProcessingResult(edits, stats)


# 【guohx】从项目根目录列表生成编辑数据集的函数，负责批量处理多个代码仓库并提取编辑信息
def dataset_from_projects(
    cache: PickleCache,  # 【guohx】缓存对象，用于加速重复处理
    project_roots: Sequence[Path],  # 【guohx】项目根目录路径列表，每个路径对应一个代码仓库
    change_processor: ProjectChangeProcessor[TProb],  # 【guohx】变更处理器，负责从Git变更生成问题对象
    repo_training: Sequence[bool],  # 【guohx】每个仓库是否为训练集的标志列表，与project_roots一一对应
    max_history_per_repo: int,  # 【guohx】每个仓库最多处理的commit数量，避免处理过长的历史
    time_limit_per_commit: float,  # 【guohx】每个commit的处理时间限制（秒）
    workers: int = DefaultWorkers,  # 【guohx】并行处理的worker数量，默认为系统CPU核心数
) -> "Mapping[Path, Sequence[TProb]]":  # 【guohx】返回项目路径到编辑列表的映射字典
    """
    Create a TokenizedEditDataset from a list of project roots and a given encoder.
    Args:
        - max_history_per_repo (int, optional): When the repo history is longer than
        this value, only the oldest portion is going to be used. Defaults to 1000.
    """
    # 【guohx】获取当前进程ID，用于创建进程特定的工作目录
    pid = os.getpid()
    # 【guohx】创建工作目录：临时目录/dataset_from_projects/pid-进程ID
    workdir = Path(tempfile.gettempdir()) / "dataset_from_projects" / f"pid-{pid}"

    # 【guohx】保存项目根目录列表到局部变量
    roots = project_roots
    # 【guohx】为每个项目创建工作子目录：workdir/repo-索引号
    workdirs = [workdir / f"repo-{i}" for i in range(len(roots))]
    try:
        # 【guohx】使用并行映射处理所有项目，每个项目调用_process_commits函数
        presults = pmap(
            _process_commits,  # 【guohx】处理单个项目的函数
            roots,  # 【guohx】项目根目录列表
            workdirs,  # 【guohx】对应的工作目录列表
            repo_training,  # 【guohx】训练标志列表
            key_args={  # 【guohx】传递给_process_commits的额外参数
                "max_history_per_repo": max_history_per_repo,  # 【guohx】每个仓库的最大历史长度
                "change_processor": change_processor,  # 【guohx】变更处理器
                "time_limit_per_commit": time_limit_per_commit,  # 【guohx】commit处理时间限制
                "cache": cache,  # 【guohx】缓存对象
            },
            max_workers=workers,  # 【guohx】最大并行worker数
            tqdm_args={"unit": "repo"},  # 【guohx】进度条显示单位
        )
    finally:
        # 【guohx】无论处理成功还是失败，都要清理工作目录
        if workdir.exists():
            print("Removing workdir:", workdir)
            shutil.rmtree(workdir)  # 【guohx】递归删除工作目录及其内容

    # 【guohx】初始化项目到编辑列表的映射字典
    project2edits = dict[Path, list[TProb]]()

    try:
        # 【guohx】初始化统计信息字典
        stats = dict[str, Any]()
        # 【guohx】遍历每个项目及其处理结果
        for root, pr in zip(roots, presults):
            # 【guohx】将项目根目录作为键，编辑列表作为值存储到映射字典中
            project2edits.setdefault(root, []).extend(pr.edits)
            # 【guohx】将处理结果的统计信息合并到总统计中
            rec_add_dict_to(stats, pr.stats)

        # 【guohx】如果存在时间日志统计信息，转换为DataFrame并显示
        if "tlogger" in stats:
            df = TimeLogger.times_to_dataframe(stats.pop("tlogger"))
            if not df.empty:
                print("Time stats:")
                display(df)
        # 【guohx】如果存在分析器错误信息，过滤掉已知错误并显示剩余错误
        if "analyzer_errors" in list(stats.keys()):
            errors: dict = stats.pop("analyzer_errors")
            for k in list(errors.keys()):
                if JediUsageAnalyzer.is_known_error(k):  # 【guohx】过滤掉已知的分析器错误
                    errors.pop(k)
            if errors:  # 【guohx】如果还有未知错误，按错误数量排序显示
                print("Analyzer errors:")
                for k in sorted(errors.keys(), key=lambda k: errors[k], reverse=True):
                    print(f"{k}:\t{errors[k]}")
        # 【guohx】如果还有其他统计信息，使用pretty_print_dict格式化显示
        if stats:
            print("Other Stats:")
            pretty_print_dict(stats)
    except Exception as e:
        # 【guohx】如果统计信息处理过程中出现错误（非键盘中断），打印错误信息
        if not isinstance(e, KeyboardInterrupt):
            print("Error while printing stats:", e)

    # 【guohx】返回项目路径到编辑列表的映射字典
    return project2edits


# 【guohx】从仓库分集中生成数据集的函数，负责组织和管理不同分集（train/valid/test）的仓库处理
# 【guohx】该函数会遍历指定数据集下的各个分集目录，收集仓库路径，并调用dataset_from_projects进行批量处理
def datasets_from_repo_splits(
    cache: PickleCache,  # 【guohx】缓存对象，用于加速重复处理
    repos_root: Path,  # 【guohx】原始仓库根目录路径
    change_processor: ProjectChangeProcessor[TProb],  # 【guohx】变更处理器，负责从Git变更生成问题对象
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
    debug_log_path = Path("datasets_from_repo_splits_debug.log")
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
        with open(debug_log_path, "a", encoding="utf-8") as f:
                for split, ps in projects.items():
                    f.write(f"[DEBUG] split={split}, projects={[str(p) for p in ps]}\n")
    # 【guohx】调用dataset_from_projects进行批量处理
    # 【guohx】将所有分集的项目路径合并，统一处理，提高效率
    dataset = dataset_from_projects(
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


class C3ProblemDataset(TypedDict, Generic[TProb]):
    train: Sequence[TProb]
    valid: Sequence[TProb]
    test: Sequence[TProb]


# 【guohx】生成或加载数据集的核心函数，负责从原始代码仓库生成结构化的编辑问题数据集
# 【guohx】该函数会遍历指定数据集下的仓库，提取Git提交历史，生成C3Problem对象，并缓存结果
def make_or_load_ts_dataset(
    dataset_name: str,  # 【guohx】数据集名称，如 "perm2k"，对应datasets_root下的目录
    change_processor: ProjectChangeProcessor[TProb],  # 【guohx】变更处理器，负责从Git变更生成问题对象
    splits: Sequence[str],  # 【guohx】要处理的数据分集，如 ("train", "valid", "test")
    remake_problems: bool = False,  # 【guohx】是否强制重新生成问题，False时会优先使用缓存
    time_limit_per_commit: float = 10.0,  # 【guohx】每个commit的处理时间限制（秒）
    workers: int = DefaultWorkers,  # 【guohx】并行处理的worker数量
) -> C3ProblemDataset[TProb]:  # 【guohx】返回包含train/valid/test分集的数据集
    # 【guohx】生成问题处理器的配置字符串，用于缓存目录命名
    prob_config = repr_modified_args(change_processor)
    # 【guohx】使用哈希值作为目录名，避免路径过长或特殊字符问题
    import hashlib
    prob_config_hash = hashlib.md5(prob_config.encode()).hexdigest()[:16]
    # 【guohx】构建处理后的数据存储目录：datasets_root/dataset_name/processed/配置哈希值
    processed_dir = get_dataset_dir(dataset_name) / "processed"
    cache_dir = processed_dir / prob_config_hash
    # 【guohx】初始化缓存对象，用于存储/读取处理结果
    cache = PickleCache(cache_dir)
    
    # 【guohx】如果需要重新生成，清空缓存
    if remake_problems:
        cache.clear()
    
    # 【guohx】调用核心处理函数，从仓库分集中生成问题数据集
    # 【guohx】datasets_from_repo_splits会：
    #   - 遍历每个分集目录下的仓库
    #   - 对每个仓库调用edits_from_commit_history提取变更
    #   - 用change_processor将变更转换为问题对象
    #   - 返回按分集组织的问题列表
    results = datasets_from_repo_splits(
        cache,  # 【guohx】缓存对象，用于加速重复处理
        get_dataset_dir(dataset_name) / "repos",  # 【guohx】原始仓库根目录
        change_processor,  # 【guohx】变更处理器
        workers=workers,  # 【guohx】并行worker数
        splits=splits,  # 【guohx】要处理的分集
        time_limit_per_commit=time_limit_per_commit,  # 【guohx】commit处理时间限制
    )
    
    # 【guohx】统计缓存文件的总大小，用于监控存储使用情况
    size_mb = 0.0
    n = 0
    for f in cache_dir.iterdir():
        n += 1
        size_mb += f.stat().st_size / (1024**2)
    print(f"Dataset total size ({n=}): {size_mb:.2f} MB")

    # 【guohx】构造并返回C3ProblemDataset对象，包含train/valid/test三个分集
    # 【guohx】每个分集包含该分集下所有仓库生成的问题对象列表
    return C3ProblemDataset(
        train=results.get("train", []),  # 【guohx】训练集问题列表
        valid=results.get("valid", []),  # 【guohx】验证集问题列表
        test=results.get("test", []),    # 【guohx】测试集问题列表
    )


def make_or_load_ts_transformed_dataset(
    dataset_name: str,
    dataset: C3ProblemDataset | None,
    encoder: TsC3CombinedEncoder,
    remake_problems: bool = False,
    workers: int = DefaultWorkers,
) -> dict[str, Sequence[TsC3Problem]]:
    def transform_eval_problems(
        dataset: C3ProblemDataset,
    ) -> dict[str, Sequence[TsC3Problem]]:
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
    # 使用哈希值作为缓存键，避免路径过长或特殊字符问题
    import hashlib
    cache_key = f"eval-{hashlib.md5(proc_config.encode()).hexdigest()[:8]}-{hashlib.md5(trans_config.encode()).hexdigest()[:8]}"
    transformed_dir = get_dataset_dir(dataset_name) / "transformed"
    cache = PickleCache(transformed_dir)
    return cache.cached(
        cache_key,
        lambda: transform_eval_problems(not_none(dataset)),
        remake=remake_problems,
    )


def save_datasets(datasets: Mapping[str, Any], save_dir: Path) -> None:
    for name, dataset in datasets.items():
        pickle_dump(save_dir / f"{name}.pkl", dataset)
    subprocess.run(["du", "-sh", save_dir])


def load_datasets(save_dir: Path, splits=("test", "valid", "train")) -> dict[str, Any]:
    return {
        name: pickle_load(path)
        for name in splits
        if (path := (save_dir / f"{name}.pkl")).exists()
    }


def get_repo_signature(repo: Path, n_commits: int = 30) -> tuple[str, ...]:
    # use the first n commits as the signature
    commits = get_commit_history(repo)[-n_commits:]
    return tuple(c.msg for c in commits)
