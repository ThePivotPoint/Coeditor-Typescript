import copy
import gc
import shutil
import sys
import time
import warnings
from abc import ABC, abstractmethod
from functools import cached_property

import jedi
import jedi.settings
import parso
from jedi.file_io import FileIO, FolderIO
from jedi.inference.context import ModuleContext
from jedi.inference.references import recurse_find_python_files
from parso.python import tree as ptree
from parso.tree import BaseNode, NodeOrLeaf

from ._utils import rec_iter_files
from .change import Added, Change, Deleted, Modified, get_named_changes
from .common import *
from .encoding import change_to_line_diffs, line_diffs_to_original_delta
from .git import CommitInfo

ScopeTree = ptree.Function | ptree.Class | ptree.Module
PyNode = ptree.PythonBaseNode | ptree.PythonNode


class LineRange(NamedTuple):
    start: int
    until: int

    def __contains__(self, l: int) -> bool:
        return self.start <= l < self.until

    def to_range(self) -> range:
        return range(self.start, self.until)


_tlogger = TimeLogger()


def line_range(start: int, end: int, can_be_empty: bool = False) -> LineRange:
    if not can_be_empty and start >= end:
        raise ValueError(f"Bad line range: {start=}, {end=}")
    return LineRange(start, end)


def _strip_empty_lines(s: str):
    s1 = s.lstrip("\n")
    s2 = s1.rstrip("\n")
    e_lines_left = len(s) - len(s1)
    e_lines_right = len(s1) - len(s2)
    return s2, e_lines_left, e_lines_right


@dataclass
class ChangeScope:
    """
    A change scope is a python module, non-hidden function, or a non-hidden class, or a python module.
        - functions and classes that are inside a parent function are considered hidden.
    """

    path: ProjectPath
    tree: ScopeTree
    spans: Sequence["StatementSpan"]
    subscopes: Mapping[str, Self]
    parent_scope: "ChangeScope | None"

    def __post_init__(self):
        # 计算头文件
        if isinstance(self.tree, ptree.Module):
            header_code = f"# module: {self.path.module}"
            header_line_range = line_range(0, 0, can_be_empty=True)
        else:
            h_start, h_end = 0, 0
            tree = self.tree
            to_visit = list[NodeOrLeaf]()
            parent = not_none(tree.parent)
            while parent.type in ("decorated", "async_funcdef"):
                to_visit.insert(0, parent.children[0])
                parent = not_none(parent.parent)
            to_visit.extend(tree.children)
            visited = list[NodeOrLeaf]()
            for c in to_visit:
                if c.type == "suite":
                    break
                visited.append(c)
            header_code = "".join(cast(str, c.get_code()) for c in visited)
            header_code, e_left, e_right = _strip_empty_lines(header_code)
            h_start = not_none(visited[0].get_start_pos_of_prefix())[0] + e_left
            h_end = visited[-1].end_pos[0] + 1 - e_right
            # assert_eq(count_lines(header_code), h_end - h_start)
            header_line_range = line_range(h_start, h_end)
            if self.spans and h_end > self.spans[0].line_range[0]:
                raise ValueError(
                    f"Header covers the fisrt span: {self.path=}, {h_start=}, {h_end=} "
                    f"{self.spans[0].line_range=}"
                )

        self.header_code: str = header_code + "\n"
        self.header_line_range: LineRange = header_line_range

    def ancestors(self) -> list[Self]:
        scope = self
        result = [scope]
        while scope := scope.parent_scope:
            result.append(scope)
        result.reverse()
        return result

    @cached_property
    def spans_code(self) -> str:
        return "\n".join(s.code for s in self.spans)

    @cached_property
    def all_code(self) -> str:
        return self.header_code + self.spans_code

    def search_span_by_line(self, line: int) -> "StatementSpan | None":
        # TODO: optimize this to avoid linear scan
        span = self._search_span(line)
        if span is not None:
            return span
        for s in self.subscopes.values():
            span = s.search_span_by_line(line)
            if span is not None:
                return span

    def _search(self, path: ElemPath, line: int) -> Self | "StatementSpan":
        scope = self._search_scope(path)
        if scope.header_line_range[0] <= line < scope.header_line_range[1]:
            return scope
        span = scope._search_span(line)
        return span or scope

    def _search_scope(self, path: ElemPath) -> Self:
        """Find the scope that can potentially contain the given path. Follow the
        path segments until no more subscopes are found."""
        segs = split_dots(path)
        scope = self
        for s in segs:
            if s in scope.subscopes:
                scope = scope.subscopes[s]
            else:
                break
        return scope

    def _search_span(self, line: int) -> "StatementSpan | None":
        for span in self.spans:
            if line in span.line_range:
                return span
        return None

    @staticmethod
    def from_tree(path: ProjectPath, tree: ScopeTree) -> "ChangeScope":
        spans = []
        subscopes = dict()
        scope = ChangeScope(path, tree, spans, subscopes, None)
        assert isinstance(tree, ScopeTree)
        is_func = isinstance(tree, ptree.Function)

        def mk_span(stmts):
            # 移除前导换行符
            n_leading_newlines = 0
            for s in stmts:
                if s.type == ptree.Newline.type:
                    n_leading_newlines += 1
                else:
                    break
            if n_leading_newlines:
                stmts = stmts[n_leading_newlines:]
            if stmts:
                yield StatementSpan(len(spans), stmts, scope)

        current_stmts = []
        container = tree if isinstance(tree, ptree.Module) else tree.get_suite()
        if isinstance(container, BaseNode):
            content = container.children
        else:
            content = []
        for s in content:
            # 对于函数内容，不创建内部作用域
            if is_func or _is_scope_statement(as_any(s)):
                current_stmts.append(s)
            else:
                if current_stmts:
                    spans.extend(mk_span(current_stmts))
                    current_stmts = []
        if current_stmts:
            spans.extend(mk_span(current_stmts))

        if is_func:
            # 对于函数内容，不创建内部作用域
            if not spans:
                raise ValueError(f"Function with no spans: {path=}, {tree.get_code()=}")
            return scope
        for stree in tree._search_in_scope(ptree.Function.type, ptree.Class.type):
            stree: ptree.Function | ptree.Class
            name = cast(ptree.Name, stree.name).value
            spath = path.append(name)
            subscope = ChangeScope.from_tree(spath, stree)
            subscope.parent_scope = scope
            subscopes[name] = subscope
        return scope

    def __repr__(self):
        return (
            f"ChangeScope(path={self.path}, type={self.tree.type}, spans={self.spans})"
        )


_non_scope_stmt_types = {
    "decorated",
    "async_stmt",
    ptree.Class.type,
    ptree.Function.type,
}


def _is_scope_statement(stmt: PyNode) -> bool:
    """Will only return False for functions, classes, and import statments"""
    if stmt.type in _non_scope_stmt_types:
        return False
    if stmt.type == "simple_stmt" and stmt.children[0].type in ptree._IMPORTS:
        return False
    return True


@dataclass
class StatementSpan:
    """
    A statement span is a set of lines inside the same change scope. It is the basic unit of code changes handled by our model.
        - For a modified function, the span is the function itself.
        - For a modified module, the spans are the regions between the functions and classes plus
        the spans recursively generated.
        - For a modified class, the spans are the regions between methods plus
        the spans recursively generated.
    """

    nth_in_parent: int
    statements: Sequence[PyNode]
    scope: ChangeScope

    def __post_init__(self):
        assert self.statements
        origin_code = "".join(s.get_code() for s in self.statements)
        code, e_left, e_right = _strip_empty_lines(origin_code)
        start = not_none(self.statements[0].get_start_pos_of_prefix())[0] + e_left
        end = self.statements[-1].end_pos[0] + 1 - e_right

        self.code: str = code + "\n"
        try:
            self.line_range: LineRange = line_range(start, end)
        except ValueError:
            print_err(f"{e_right=}, {start=}, {end=}")
            print_err("Origin code:")
            print_err(origin_code)
            print_err("Stmts:")
            for s in self.statements:
                print_err(s)
            raise

    def __repr__(self):
        preview = self.code
        str_limit = 30
        if len(preview) > str_limit:
            preview = preview[:str_limit] + "..."
        return f"StatementSpan({self.line_range}, code={repr(preview)})"


@dataclass(frozen=True)
class ChangedSpan:
    "Represents the changes made to a statement span."
    change: Change[str]
    parent_scopes: Sequence[Change[ChangeScope]]
    line_range: LineRange

    def inverse(self) -> "ChangedSpan":
        return ChangedSpan(
            self.change.inverse(),
            [c.inverse() for c in self.parent_scopes],
            self.line_range,
        )

    @property
    def header_line_range(self) -> LineRange:
        parent_scope = self.parent_scopes[-1].earlier
        hrange = parent_scope.header_line_range
        return hrange

    @property
    def module(self) -> ModuleName:
        return self.parent_scopes[-1].earlier.path.module

    @property
    def scope(self) -> Change[ChangeScope]:
        return self.parent_scopes[-1]

    def _is_func_body(self) -> bool:
        return self.parent_scopes[-1].earlier.tree.type == ptree.Function.type

    def __repr__(self) -> str:
        return f"ChangeSpan(module={self.module}, range={self.line_range}, scope={self.scope.earlier.path.path}, type={self.change.as_char()})"


@dataclass
class JModule:
    "A light wrapper around a jedi module."
    mname: ModuleName
    tree: ptree.Module

    @cached_property
    def as_scope(self) -> ChangeScope:
        return ChangeScope.from_tree(ProjectPath(self.mname, ""), self.tree)

    @cached_property
    def imported_names(self):
        names = set[ptree.Name]()
        for stmt in self.tree.iter_imports():
            if isinstance(stmt, ptree.ImportFrom):
                for n in stmt.get_from_names():
                    assert isinstance(n, ptree.Name)
                    names.add(n)
            elif isinstance(stmt, ptree.ImportName):
                for n in stmt.get_defined_names():
                    assert isinstance(n, ptree.Name)
                    names.add(n)
        return names


@dataclass(frozen=True)
class JModuleChange:
    module_change: Change[JModule]
    changed: Sequence[ChangedSpan]

    def __repr__(self) -> str:
        return f"JModuleChange({self.changed})"

    def inverse(self) -> Self:
        "Create the inverse change."
        return JModuleChange(
            self.module_change.inverse(), [span.inverse() for span in self.changed]
        )

    @staticmethod
    def from_modules(
        module_change: Change[JModule],
        only_ast_changes: bool = True,
        return_unchanged: bool = False,
    ):
        "Compute the change spans from two versions of the same module."
        with _tlogger.timed("JModuleChange.from_modules"):
            changed = get_changed_spans(
                module_change.map(lambda m: m.as_scope),
                tuple(),
                only_ast_changes=only_ast_changes,
                return_unchanged=return_unchanged,
            )
            return JModuleChange(module_change, changed)


def get_python_files(project: Path) -> list[RelPath]:
    files = list[RelPath]()
    for f in recurse_find_python_files(FolderIO(str(project))):
        f: FileIO
        files.append(to_rel_path(Path(f.path).relative_to(project)))
    return files


DefaultIgnoreDirs = {".venv", ".mypy_cache", ".git", "venv", "build"}


@dataclass(frozen=True)
class EditTarget:
    lines: tuple[int, int]


@dataclass(frozen=True)
class JProjectChange:
    project_name: str
    changed: Mapping[ModuleName, JModuleChange]
    all_modules: Modified[Collection[JModule]]
    commit_info: "CommitInfo | None"

    def __repr__(self) -> str:
        commit = (
            f"commit={repr(self.commit_info.summary())}, " if self.commit_info else ""
        )
        return f"JProjectChange({commit}{self.changed})"


@dataclass
class ProjectState:
    project: jedi.Project
    scripts: Mapping[RelPath, jedi.Script]


TProb = TypeVar("TProb", covariant=True)
TEnc = TypeVar("TEnc", covariant=True)


class ProjectChangeProcessor(Generic[TProb], ABC):
    def pre_edit_analysis(
        self,
        pstate: ProjectState,
        modules: Mapping[RelPath, JModule],
        changes: Mapping[ModuleName, JModuleChange],
    ) -> Any:
        return None

    def post_edit_analysis(
        self,
        pstate: ProjectState,
        modules: Mapping[RelPath, JModule],
        changes: Mapping[ModuleName, JModuleChange],
    ) -> Any:
        return None

    @abstractmethod
    def process_change(
        self, pchange: "JProjectChange", pre_analysis: Any, post_analysis: Any
    ) -> Sequence[TProb]:
        ...

    def clear_stats(self):
        return None

    def append_stats(self, stats: dict[str, Any]) -> None:
        return None

    def set_training(self, is_training: bool) -> None:
        self._is_training = is_training

    @property
    def is_training(self) -> bool:
        return getattr(self, "_is_training", False)

    def use_unchanged(self) -> bool:
        return False

    @staticmethod
    def should_mk_problem(
        span: ChangedSpan, func_only: bool, max_chars: int, max_lines: int
    ):
        return (
            (span.change.as_char() == Modified.as_char())
            and (not func_only or span._is_func_body())
            and (len(span.change.earlier) <= max_chars)
            and (len(span.change.later) <= max_chars)
            and (count_lines(span.change.earlier) <= max_lines)
            and (count_lines(span.change.later) <= max_lines)
        )


class NoProcessing(ProjectChangeProcessor[JProjectChange]):
    def process_change(
        self,
        pchange: JProjectChange,
        pre_analysis,
        post_analysis,
    ) -> Sequence[JProjectChange]:
        return [pchange]


# 【guohx】从Git提交历史中增量计算项目编辑信息的主入口函数
def edits_from_commit_history(
    project_dir: Path,  # 【guohx】原始项目目录路径
    history: Sequence[CommitInfo],  # 【guohx】要处理的提交历史列表，按时间顺序排列
    tempdir: Path,  # 【guohx】临时工作目录路径，用于存放项目副本
    change_processor: ProjectChangeProcessor[TProb] = NoProcessing(),  # 【guohx】变更处理器，负责生成问题对象，默认为无处理
    ignore_dirs=DefaultIgnoreDirs,  # 【guohx】要忽略的目录集合，默认为系统默认忽略目录
    silent: bool = False,  # 【guohx】是否静默模式，控制进度条显示
    time_limit: float | None = None,  # 【guohx】处理时间限制（秒），None表示无限制
) -> Sequence[TProb]:  # 【guohx】返回生成的问题对象序列
    """Incrementally compute the edits to a project from the git history.
    Note that this will change the file states in the project directory, so
    you should make a copy of the project before calling this function.
    """
    # 【guohx】解析临时目录的绝对路径，确保路径正确
    tempdir = tempdir.resolve()
    # 【guohx】检查临时目录是否已存在，存在则抛出异常避免覆盖
    if tempdir.exists():
        raise FileExistsError(f"Workdir '{tempdir}' already exists.")
    # 【guohx】保存当前Jedi快速解析器设置，用于后续恢复
    use_fast_parser = jedi.settings.fast_parser
    # 【guohx】创建临时目录，parents=True创建父目录，exist_ok=False确保目录不存在
    tempdir.mkdir(parents=True, exist_ok=False)
    try:
        # 【guohx】复制原始项目的.git目录到临时目录，保持Git历史信息
        run_command(
            ["cp", "-r", str(project_dir / ".git"), str(tempdir)],  # 【guohx】递归复制.git目录
            cwd=project_dir.parent,  # 【guohx】在项目父目录下执行命令
        )

        # 【guohx】调用内部函数进行实际的提交历史处理
        return _edits_from_commit_history(
            tempdir,  # 【guohx】临时工作目录
            history,  # 【guohx】提交历史
            change_processor,  # 【guohx】变更处理器
            ignore_dirs,  # 【guohx】忽略目录
            silent,  # 【guohx】静默模式
            time_limit=time_limit,  # 【guohx】时间限制
        )
    finally:
        # 【guohx】无论处理成功还是失败，都要清理临时目录
        shutil.rmtree(tempdir)  # 【guohx】递归删除临时目录及其内容
        # 【guohx】恢复Jedi快速解析器设置
        jedi.settings.fast_parser = use_fast_parser
        # 【guohx】强制垃圾回收，释放内存
        gc.collect()


def _deep_copy_subset_(dict: dict[T1, T2], keys: Collection[T1]) -> dict[T1, T2]:
    "This is more efficient than deepcopying each value individually if they share common data."
    keys = {k for k in keys if k in dict}
    to_copy = {k: dict[k] for k in keys}
    copies = copy.deepcopy(to_copy)
    for k in keys:
        dict[k] = copies[k]
    return dict


_Second = float


def parse_module_script(project: jedi.Project, path: Path):
    assert path.is_absolute(), f"Path is not absolute: {path=}"
    script = jedi.Script(path=path, project=project)
    mcontext = script._get_module_context()
    assert isinstance(mcontext, ModuleContext)
    mname = cast(str, mcontext.py__name__())
    if mname.startswith("src."):
        e = ValueError(f"Bad module name: {mname}")
        files = list(project.path.iterdir())
        print_err(f"project: {project.path}", file=sys.stderr)
        print_err(f"files in root: {files}", file=sys.stderr)
        raise e
    m = script._module_node
    assert isinstance(m, ptree.Module)
    # mname = PythonProject.rel_path_to_module_name(path.relative_to(proj.path))
    # m = parso.parse(path.read_text())
    jmod = JModule(mname, m)
    return jmod, script


# 【guohx】从Git提交历史中增量计算编辑信息的核心内部函数，实现时间倒序遍历和模块变更分析
def _edits_from_commit_history(
    project: Path,  # 【guohx】项目工作目录路径（临时目录）
    history: Sequence[CommitInfo],  # 【guohx】提交历史列表，按时间顺序排列
    change_processor: ProjectChangeProcessor[TProb],  # 【guohx】变更处理器，负责生成问题对象
    ignore_dirs: set[str],  # 【guohx】要忽略的目录集合
    silent: bool,  # 【guohx】是否静默模式，控制进度条显示
    time_limit: _Second | None,  # 【guohx】处理时间限制（秒），None表示无限制
) -> Sequence[TProb]:  # 【guohx】返回生成的问题对象序列
    # 【guohx】记录处理开始时间，用于超时检查
    start_time = time.time()
    # 【guohx】初始化Jedi脚本缓存字典，键为相对路径，值为Jedi脚本对象
    scripts = dict[RelPath, jedi.Script]()
    # 【guohx】初始化结果列表，存储生成的问题对象
    results = list[TProb]()

    # 【guohx】检查是否超时的内部函数
    def has_timeouted(step):
        if time_limit and (time.time() - start_time > time_limit):  # 【guohx】如果设置了时间限制且已超时
            warnings.warn(  # 【guohx】输出超时警告信息
                f"_edits_from_commit_history timed out for {project}. ({time_limit=}) "
                f"Partial results ({step}/{len(history)-1}) will be returned."
            )
            return True
        else:
            return False

    # 【guohx】解析Python模块的内部函数，使用Jedi进行语法分析
    def parse_module(path: Path):
        with _tlogger.timed("parse_module"):  # 【guohx】记录模块解析时间
            m, s = parse_module_script(proj, path)  # 【guohx】解析模块并获取JModule和Script对象
            scripts[to_rel_path(path.relative_to(proj._path))] = s  # 【guohx】将Script对象缓存到scripts字典
            return m  # 【guohx】返回JModule对象

    # 【guohx】切换到指定提交的内部函数
    def checkout_commit(commit_hash: str):
        with _tlogger.timed("checkout"):  # 【guohx】记录Git checkout时间
            subprocess.run(  # 【guohx】执行Git checkout命令
                ["git", "checkout", "-f", commit_hash],  # 【guohx】强制切换到指定提交
                cwd=project,  # 【guohx】在项目目录下执行
                capture_output=True,  # 【guohx】捕获输出
                check=True,  # 【guohx】检查命令是否成功
            )

    # 【guohx】确保工作目录只包含.git目录，避免意外覆盖真实代码
    if list(project.iterdir()) != [project / ".git"]:
        raise FileExistsError(f"Directory '{project}' should contain only '.git'.")

    # 【guohx】切换到历史中的第一个提交（最新的提交）
    commit_now = history[-1]  # 【guohx】获取最新的提交
    checkout_commit(commit_now.hash)  # 【guohx】切换到最新提交
    # 【guohx】创建Jedi项目对象，添加src目录到Python路径
    proj = jedi.Project(path=project, added_sys_path=[project / "src"])
    # 【guohx】创建项目状态对象，包含项目和脚本信息
    pstate = ProjectState(proj, scripts)

    # 【guohx】现在我们可以获取第一个项目状态，虽然现在不需要
    # 【guohx】但稍后会用于预编辑分析
    # 【guohx】收集初始的Python源文件列表
    init_srcs = [
        to_rel_path(f.relative_to(project))  # 【guohx】转换为相对路径
        for f in rec_iter_files(project, dir_filter=lambda d: d.name not in ignore_dirs)  # 【guohx】递归遍历文件，过滤忽略目录
        if f.suffix == ".py" and (project / f).exists()  # 【guohx】只处理存在的Python文件
    ]
    # 【guohx】构建路径到模块的映射字典，解析所有初始源文件
    path2module = {
        f: parse_module(project / f)  # 【guohx】解析每个Python文件为JModule对象
        for f in tqdm(init_srcs, desc="building initial project", disable=silent)  # 【guohx】显示进度条
    }

    # 【guohx】判断文件是否为源文件的内部函数
    def is_src(path_s: str) -> bool:
        path = Path(path_s)
        return path.suffix == ".py" and all(p not in ignore_dirs for p in path.parts)  # 【guohx】检查是否为Python文件且不在忽略目录中

    # 【guohx】获取未来要处理的提交列表（时间倒序，从旧到新）
    future_commits = list(reversed(history[:-1]))
    # 【guohx】遍历每个提交，进行增量编辑分析
    for step, commit_next in tqdm(
        list(enumerate(future_commits)),  # 【guohx】枚举未来提交
        smoothing=0,  # 【guohx】进度条平滑参数
        desc="processing commits",  # 【guohx】进度条描述
        disable=silent,  # 【guohx】根据静默模式控制进度条
    ):
        # 【guohx】检查是否超时
        if has_timeouted(step):
            return results
        # 【guohx】获取当前提交和下一个提交之间的文件变更
        changed_files = run_command(  # 【guohx】执行Git diff命令
            [
                "git",
                "diff",
                "--no-renames",  # 【guohx】不检测重命名
                "--name-status",  # 【guohx】只输出文件名和状态
                commit_now.hash,  # 【guohx】当前提交
                commit_next.hash,  # 【guohx】下一个提交
            ],
            cwd=project,  # 【guohx】在项目目录下执行
        ).splitlines()  # 【guohx】按行分割输出

        # 【guohx】初始化路径变更集合
        path_changes = set[Change[str]]()

        # 【guohx】解析Git diff输出，识别文件变更类型
        for line in changed_files:
            segs = line.split("\t")  # 【guohx】按制表符分割
            if len(segs) == 2:  # 【guohx】两个字段：状态和路径
                tag, path = segs
                if not is_src(path):  # 【guohx】如果不是源文件，跳过
                    continue
                if tag.endswith("A"):  # 【guohx】添加的文件
                    path_changes.add(Added(path))
                elif tag.endswith("D"):  # 【guohx】删除的文件
                    path_changes.add(Deleted(path))
                if tag.endswith("M"):  # 【guohx】修改的文件
                    path_changes.add(Modified(path, path))
            elif len(segs) == 3:  # 【guohx】三个字段：状态、旧路径、新路径（重命名）
                tag, path1, path2 = segs
                assert tag.startswith("R")  # 【guohx】确保是重命名操作
                if is_src(path1):  # 【guohx】如果旧路径是源文件
                    path_changes.add(Deleted(path1))
                if is_src(path2):  # 【guohx】如果新路径是源文件
                    path_changes.add(Added(path2))

        # 【guohx】深拷贝将要更改的模块，保存修改前的状态
        to_copy = {
            to_rel_path(Path(path_change.before))  # 【guohx】获取变更前的路径
            for path_change in path_changes
            if not isinstance(path_change, Added)  # 【guohx】只复制非新增的模块
        }
        _deep_copy_subset_(path2module, to_copy)  # 【guohx】深拷贝指定的模块

        # 【guohx】切换到下一个提交
        checkout_commit(commit_next.hash)

        # 【guohx】复制模块映射，准备更新
        new_path2module = path2module.copy()
        # 【guohx】初始化变更字典，存储模块级别的变更
        changed = dict[ModuleName, JModuleChange]()
        # 【guohx】处理每个路径变更
        for path_change in path_changes:
            path = project / path_change.earlier  # 【guohx】获取变更前的路径
            rel_path = to_rel_path(path.relative_to(project))  # 【guohx】转换为相对路径
            # 【guohx】如果非新增变更且模块不存在，处理异常情况
            if not isinstance(path_change, Added) and rel_path not in new_path2module:
                warnings.warn(f"No module for file: {project/rel_path}")
                if isinstance(path_change, Deleted):  # 【guohx】如果是删除，跳过
                    continue
                elif isinstance(path_change, Modified):  # 【guohx】如果是修改，转换为新增
                    path_change = Added(path_change.after)
            # 【guohx】根据变更类型进行相应处理
            match path_change:
                case Added():  # 【guohx】处理新增文件
                    mod = parse_module(path)  # 【guohx】解析新模块
                    new_path2module[rel_path] = mod  # 【guohx】添加到模块映射
                    changed[mod.mname] = JModuleChange.from_modules(Added(mod))  # 【guohx】创建模块变更
                case Deleted():  # 【guohx】处理删除文件
                    mod = new_path2module.pop(rel_path)  # 【guohx】从映射中移除模块
                    changed[mod.mname] = JModuleChange.from_modules(Deleted(mod))  # 【guohx】创建删除变更
                case Modified(path1, path2):  # 【guohx】处理修改文件
                    assert path1 == path2  # 【guohx】确保路径相同
                    mod_old = new_path2module.pop(rel_path)  # 【guohx】移除旧模块
                    new_path2module[rel_path] = mod_new = parse_module(path)  # 【guohx】解析新模块
                    changed[mod_new.mname] = JModuleChange.from_modules(  # 【guohx】创建修改变更
                        Modified(mod_old, mod_new),
                        return_unchanged=change_processor.use_unchanged(),  # 【guohx】根据处理器设置决定是否返回未变更部分
                    )
            # 【guohx】检查是否超时
            if has_timeouted(step):
                return results

        # 【guohx】创建项目级别的模块变更
        modules_mod = Modified(path2module.values(), new_path2module.values())
        # 【guohx】创建项目变更对象
        pchange = JProjectChange(project.name, changed, modules_mod, commit_next)

        # 【guohx】执行后编辑分析
        with _tlogger.timed("post_edit_analysis"):  # 【guohx】记录后编辑分析时间
            post_analysis = change_processor.post_edit_analysis(  # 【guohx】调用处理器的后编辑分析
                pstate,  # 【guohx】项目状态
                new_path2module,  # 【guohx】新的模块映射
                changed,  # 【guohx】变更信息
            )
        # 【guohx】检查是否超时
        if has_timeouted(step):
            return results

        # 【guohx】现在向后遍历时间以执行预编辑分析
        checkout_commit(commit_now.hash)  # 【guohx】切换回当前提交
        # 【guohx】执行预编辑分析
        with _tlogger.timed("pre_edit_analysis"):  # 【guohx】记录预编辑分析时间
            pre_analysis = change_processor.pre_edit_analysis(  # 【guohx】调用处理器的预编辑分析
                pstate,  # 【guohx】项目状态
                path2module,  # 【guohx】当前模块映射
                changed,  # 【guohx】变更信息
            )
        # 【guohx】检查是否超时
        if has_timeouted(step):
            return results
        # 【guohx】切换回下一个提交，准备处理
        checkout_commit(commit_next.hash)

        # 【guohx】执行变更处理，生成问题对象
        with _tlogger.timed("process_change"):  # 【guohx】记录变更处理时间
            processed = change_processor.process_change(  # 【guohx】调用处理器的变更处理
                pchange, pre_analysis, post_analysis
            )
            results.extend(processed)
        commit_now = commit_next
        path2module = new_path2module
    return results


def get_changed_spans(
    scope_change: Change[ChangeScope],
    parent_changes: tuple[Change[ChangeScope], ...] = (),
    only_ast_changes: bool = True,
    return_unchanged: bool = False,
) -> list[ChangedSpan]:
    """
    Extract the change spans from scope change.
        - We need a tree differencing algorithm that are robust to element movements.
        - To compute the changes to each statement region, we can compute the differences
        by concatenating all the regions before and after the edit
        (and hiding all the sub spans such as class methods), then map the changes
        to each line back to the original regions.

    ## Args:
    - `only_ast_changes`: if True, will skip the changes that are just caused by
    comments or formatting changes.
    - `return_unchanged`: if True, unchanged code spans will also be returned as
    ChangedSpan.
    """

    def get_modified_spans(
        old_scope: ChangeScope,
        new_scope: ChangeScope,
        parent_changes: Sequence[Change[ChangeScope]],
    ) -> Iterable[ChangedSpan]:
        if (
            not return_unchanged
            and only_ast_changes
            and code_equal(old_scope.spans_code, new_scope.spans_code)
        ):
            return
        diffs = change_to_line_diffs(
            Modified(old_scope.spans_code, new_scope.spans_code)
        )
        _, delta = line_diffs_to_original_delta(diffs)
        line = 0
        for span in old_scope.spans:
            code = span.code
            line_range = (line, line + count_lines(code))
            subdelta = delta.for_input_range(line_range).shifted(-line)
            if subdelta:
                new_code = subdelta.apply_to_input(code)
                change = Modified(code, new_code)
                yield ChangedSpan(
                    change,
                    parent_changes,
                    span.line_range,
                )
            elif return_unchanged:
                yield ChangedSpan(
                    Modified.from_unchanged(code),
                    parent_changes,
                    span.line_range,
                )
            line = line_range[1]

    def recurse(
        scope_change: Change[ChangeScope], parent_changes
    ) -> Iterable[ChangedSpan]:
        parent_changes = (*parent_changes, scope_change)
        match scope_change:
            case Modified(old_scope, new_scope):
                # 计算语句差异
                yield from get_modified_spans(old_scope, new_scope, parent_changes)
                for sub_change in get_named_changes(
                    old_scope.subscopes, new_scope.subscopes
                ).values():
                    yield from recurse(sub_change, parent_changes)
            case Added(scope) | Deleted(scope):
                for span in scope.spans:
                    code_change = scope_change.new_value(span.code)
                    yield ChangedSpan(
                        code_change,
                        parent_changes,
                        span.line_range,
                    )
                for s in scope.subscopes.values():
                    s_change = scope_change.new_value(s)
                    yield from recurse(s_change, parent_changes)

    spans = list(recurse(scope_change, parent_changes))
    spans.sort(key=lambda s: s.line_range[0])
    return spans


def code_to_module(code: str) -> ptree.Module:
    return parso.parse(code)
