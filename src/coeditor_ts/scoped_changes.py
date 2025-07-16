"""
TypeScript scoped changes analysis module.
"""

import copy
import gc
import shutil
import sys
import time
import warnings
from abc import ABC, abstractmethod
from functools import cached_property
from typing import *

import tree_sitter
import tree_sitter_typescript

from coeditor._utils import rec_iter_files
from coeditor.change import Added, Change, Deleted, Modified, get_named_changes
from coeditor.common import *
from coeditor.encoding import change_to_line_diffs, line_diffs_to_original_delta
from coeditor.git import CommitInfo

from .common import *

# Import LineRange from the main coeditor module
from coeditor.scoped_changes import LineRange, line_range

# TypeScript non-scope statement types
_ts_non_scope_stmt_types = {
    "decorator",
    "export_statement", 
    "import_statement",
    "function_declaration",
    "class_declaration",
    "method_definition",
    "interface_declaration",
    "type_alias_declaration",
    "enum_declaration",
    "namespace_declaration",
}

def _is_ts_scope_statement(stmt: TsNode) -> bool:
    """Will only return False for functions, classes, imports, exports, etc."""
    if stmt.type in _ts_non_scope_stmt_types:
        return False
    # TypeScript specific: check for import/export statements
    if stmt.type in ["import_statement", "export_statement"]:
        return False
    return True

def _strip_empty_lines(s: str):
    """Remove leading and trailing empty lines from a string."""
    s1 = s.lstrip("\n")
    s2 = s1.rstrip("\n")
    e_lines_left = len(s) - len(s1)
    e_lines_right = len(s1) - len(s2)
    return s2, e_lines_left, e_lines_right

@dataclass
class TsChangeScope:
    """A change scope for TypeScript code."""
    path: TsProjectPath
    tree: TsScopeTree
    spans: Sequence["TsStatementSpan"]
    subscopes: Mapping[str, "TsChangeScope"]
    parent_scope: "TsChangeScope | None"

    def __post_init__(self):
        # 计算头文件
        if self.tree.type == "program":
            header_code = f"// module: {self.path}"
            header_line_range = line_range(0, 0, can_be_empty=True)
        else:
            h_start, h_end = 0, 0
            tree = self.tree
            
            # 获取函数/类的头部代码
            header_code = ""
            if tree.type in ["function_declaration", "class_declaration", "method_definition"]:
                # 获取函数/类声明到第一个大括号之前的所有代码
                start_pos = tree.start_point
                end_pos = tree.end_point
                
                # 找到函数体/类体的开始位置
                body_node = None
                for child in tree.children:
                    if child.type in ["block", "statement_block"]:
                        body_node = child
                        break
                
                if body_node:
                    # 头部代码是从开始到函数体开始
                    header_end = body_node.start_point[0]
                    # 这里需要从原始代码中提取，暂时用简化版本
                    header_code = f"// {tree.type}: {(tree.text or b'').decode('utf-8')[:100]}..."
                    h_start = start_pos[0]
                    h_end = header_end
                else:
                    header_code = f"// {tree.type}"
                    h_start = start_pos[0]
                    h_end = end_pos[0] + 1
            else:
                header_code = f"// {tree.type}"
                h_start = tree.start_point[0]
                h_end = tree.end_point[0] + 1
            
            header_line_range = line_range(h_start, h_end, can_be_empty=True)
            
            if self.spans and h_end > self.spans[0].line_range[0]:
                raise ValueError(
                    f"Header covers the first span: {self.path=}, {h_start=}, {h_end=} "
                    f"{self.spans[0].line_range=}"
                )

        self.header_code: str = header_code + "\n"
        self.header_line_range: LineRange = header_line_range

    def ancestors(self) -> list["TsChangeScope"]:
        scope = self
        result: list[TsChangeScope] = [scope]
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

    def search_span_by_line(self, line: int) -> "TsStatementSpan | None":
        span = self._search_span(line)
        if span is not None:
            return span
        for s in self.subscopes.values():
            span = s.search_span_by_line(line)
            if span is not None:
                return span
        return None

    def _search_span(self, line: int) -> "TsStatementSpan | None":
        for span in self.spans:
            if line in span.line_range:
                return span
        return None

    @staticmethod
    def from_tree(path: TsProjectPath, tree: TsScopeTree) -> "TsChangeScope":
        spans = []
        subscopes = dict()
        scope = TsChangeScope(path, tree, spans, subscopes, None)
        
        is_func = tree.type in ["function_declaration", "method_definition"]

        def mk_span(stmts):
            # 移除前导换行符
            n_leading_newlines = 0
            for s in stmts:
                if s.type == "newline":
                    n_leading_newlines += 1
                else:
                    break
            if n_leading_newlines:
                stmts = stmts[n_leading_newlines:]
            if stmts:
                yield TsStatementSpan(len(spans), stmts, scope)

        current_stmts = []
        
        # 获取容器内容
        if tree.type == "program":
            content = tree.children
        else:
            # 查找函数体/类体
            body_node = None
            for child in tree.children:
                if child.type in ["block", "statement_block"]:
                    body_node = child
                    break
            content = body_node.children if body_node else []
        
        for s in content:
            # 对于函数内容，不创建内部作用域
            if is_func or _is_ts_scope_statement(s):
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
                raise ValueError(f"Function with no spans: {path=}")
            return scope
            
        # 查找子作用域（函数、类等）
        for child in tree.children:
            if child.type in ["function_declaration", "class_declaration", "method_definition"]:
                # 获取名称
                name = "anonymous"
                for grandchild in child.children:
                    # 兼容 identifier、name、type_identifier 等
                    if grandchild.type in ["identifier", "name", "type_identifier"]:
                        name = (grandchild.text or b'').decode('utf-8')
                        break
                spath = TsProjectPath(str(path) + "." + name)
                subscope = TsChangeScope.from_tree(spath, child)
                subscope.parent_scope = scope
                subscopes[name] = subscope
                
        return scope

    def __repr__(self):
        return (
            f"TsChangeScope(path={self.path}, type={self.tree.type}, spans={len(self.spans)})"
        )

@dataclass
class TsStatementSpan:
    """A statement span in TypeScript code."""
    nth_in_parent: int
    statements: Sequence[TsNode]
    scope: TsChangeScope

    def __post_init__(self):
        assert self.statements
        origin_code = "".join((s.text or b'').decode('utf-8') for s in self.statements)
        code, e_left, e_right = _strip_empty_lines(origin_code)
        start = self.statements[0].start_point[0] + e_left
        end = self.statements[-1].end_point[0] + 1 - e_right

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
        return f"TsStatementSpan({self.line_range}, code={repr(preview)})"

@dataclass(frozen=True)
class TsChangedSpan:
    "Represents the changes made to a TypeScript statement span."
    change: Change[str]
    parent_scopes: Sequence[Change[TsChangeScope]]
    line_range: LineRange

    def inverse(self) -> "TsChangedSpan":
        return TsChangedSpan(
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
    def module(self) -> TsModuleName:
        return TsModuleName(str(self.parent_scopes[-1].earlier.path))

    @property
    def scope(self) -> Change[TsChangeScope]:
        return self.parent_scopes[-1]

    def _is_func_body(self) -> bool:
        return self.parent_scopes[-1].earlier.tree.type in ["function_declaration", "method_definition"]

    def __repr__(self) -> str:
        return f"TsChangedSpan(module={self.module}, range={self.line_range}, scope={self.scope.earlier.path}, type={self.change.as_char()})"

@dataclass
class TsModuleChange:
    module_change: Change[TsModule]
    changed: Sequence[TsChangedSpan]

    def __repr__(self) -> str:
        return f"TsModuleChange({len(self.changed)} changes)"

    def inverse(self) -> "TsModuleChange":
        "Create the inverse change."
        return TsModuleChange(
            self.module_change.inverse(), [span.inverse() for span in self.changed]
        )

    @staticmethod
    def from_modules(
        module_change: Change[TsModule],
        only_ast_changes: bool = True,
        return_unchanged: bool = False,
    ):
        # TODO: 实现TypeScript版本的模块变更分析
        return TsModuleChange(module_change, [])
@dataclass(frozen=True)
class TsProjectChange:
    project_name: str
    changed: Mapping[ModuleName, TsModuleChange]
    all_modules: Modified[Collection[TsModule]]
    commit_info: "CommitInfo | None"

    def __repr__(self) -> str:
        commit = (
            f"commit={repr(self.commit_info.summary())}, " if self.commit_info else ""
        )
        return f"TsProjectChange({commit}{self.changed})"

def get_typescript_files(project: Path) -> list[RelPath]:
    """Get all TypeScript files in the project."""
    ts_files = []
    for pattern in ["*.ts", "*.tsx"]:
        ts_files.extend(rec_iter_files(project, lambda p: p.name.endswith(pattern[1:])))
    return ts_files

def parse_typescript_module_script(project: Path, path: Path):
    """Parse a TypeScript file and return the module and parser."""
    parser = tree_sitter.Parser()
    parser.language = tree_sitter.Language(tree_sitter_typescript.language_typescript())
    
    with open(path, 'rb') as f:
        code = f.read()
    
    tree = parser.parse(code)
    module_name = ts_path_to_module_name(to_rel_path(path.relative_to(project)))
    
    return TsModule(module_name, tree, path), parser

def code_to_ts_module(code: str) -> tree_sitter.Tree:
    """Parse TypeScript code string and return the syntax tree."""
    parser = tree_sitter.Parser()
    parser.language = tree_sitter.Language(tree_sitter_typescript.language_typescript())
    return parser.parse(code.encode('utf-8')) 

@dataclass
class TsProjectState:
    """TypeScript 项目的全局状态，包含所有已解析的模块"""
    root: Path
    modules: Mapping[str, tree_sitter.Tree ]  # key: 相对路径字符串

TProb = TypeVar("TProb", covariant=True)
TEnc = TypeVar("TEnc", covariant=True)


class ProjectChangeProcessor(Generic[TProb], ABC):
    def pre_edit_analysis(
        self,
        pstate: TsProjectState,
        modules: Mapping[RelPath, TsModule],
        changes: Mapping[ModuleName, TsModuleChange],
    ) -> Any:
        return None

    def post_edit_analysis(
        self,
        pstate: TsProjectState,
        modules: Mapping[RelPath, TsModule],
        changes: Mapping[ModuleName, TsModuleChange],
    ) -> Any:
        return None

    @abstractmethod
    def process_change(
        self, pchange: "TsProjectChange", pre_analysis: Any, post_analysis: Any
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
        span: TsChangedSpan, func_only: bool, max_chars: int, max_lines: int
    ):
        return (
            (span.change.as_char() == Modified.as_char())
            and (not func_only or span._is_func_body())
            and (len(span.change.earlier) <= max_chars)
            and (len(span.change.later) <= max_chars)
            and (count_lines(span.change.earlier) <= max_lines)
            and (count_lines(span.change.later) <= max_lines)
        )


class NoProcessing(ProjectChangeProcessor[TsProjectChange]):
    def process_change(
        self,
        pchange: TsProjectChange,
        pre_analysis,
        post_analysis,
    ) -> Sequence[TsProjectChange]:
        return [pchange]

def _edits_from_ts_commit_history(
    project: Path,  # 项目工作目录路径（临时目录）
    history: Sequence[CommitInfo],  # 提交历史列表，按时间顺序排列
    change_processor,  # 变更处理器，负责生成问题对象
    ignore_dirs,  # 要忽略的目录集合
    silent: bool,  # 是否静默模式，控制进度条显示
    time_limit: float | None,  # 处理时间限制（秒），None表示无限制
) -> Sequence:
    """
    增量式地从 TypeScript 仓库的 git 历史中提取作用域变更（scope_change）问题对象。
    """
    import subprocess
    import copy
    import warnings
    import time
    from coeditor_ts.common import get_typescript_files
    from coeditor_ts.scoped_changes import parse_typescript_module_script, TsModuleChange
    from coeditor.change import Added, Deleted, Modified

    start_time = time.time()
    results = []

    def has_timeouted(step):
        if time_limit is not None and (time.time() - start_time > time_limit):
            warnings.warn(
                f"_edits_from_ts_commit_history timed out for {project}. ({time_limit=}) "
                f"Partial results ({step}/{len(history)-1}) will be returned."
            )
            return True
        return False

    def get_modules(proj_dir):
        modules = {}
        for f in get_typescript_files(proj_dir):
            abs_path = proj_dir / f
            if abs_path.exists():
                mod, _ = parse_typescript_module_script(proj_dir, abs_path)
                modules[f] = mod
        return modules

    def checkout_commit(commit_hash):
        subprocess.run(
            ["git", "checkout", "-f", commit_hash],
            cwd=project,
            capture_output=True,
            check=True,
        )

    # 确保工作目录只包含.git目录，避免意外覆盖真实代码
    if list(project.iterdir()) != [project / ".git"]:
        raise FileExistsError(f"Directory '{project}' should contain only '.git'.")

    # 切换到历史中的第一个提交（最新的提交）
    commit_now = history[-1]
    checkout_commit(commit_now.hash)
    path2module = get_modules(project)

    # 获取未来要处理的提交列表（时间倒序，从旧到新）
    future_commits = list(reversed(history[:-1]))
    for step, commit_next in enumerate(future_commits):
        if has_timeouted(step):
            return results
        # 获取当前提交和下一个提交之间的文件变更
        changed_files = subprocess.run(
            [
                "git", "diff", "--no-renames", "--name-status",
                commit_now.hash, commit_next.hash
            ],
            cwd=project,
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

        # 深拷贝将要更改的模块，保存修改前的状态
        to_copy = {c.before for c in path_changes if hasattr(c, 'before')}
        for k in to_copy:
            if k in path2module:
                path2module[k] = copy.deepcopy(path2module[k])

        # 切换到下一个提交
        checkout_commit(commit_next.hash)
        new_path2module = get_modules(project)

        # 生成模块变更
        changed = {}
        for path_change in path_changes:
            path_str = getattr(path_change, 'earlier', None)
            if path_str is None:
                path_str = getattr(path_change, 'after', None)
            if path_str is None:
                continue
            path = project / path_str
            rel_path = path.relative_to(project)
            if isinstance(path_change, Added):
                if rel_path.exists():
                    mod, _ = parse_typescript_module_script(project, rel_path)
                    new_path2module[rel_path] = mod
                    changed[mod.mname] = TsModuleChange.from_modules(Added(mod))
            elif isinstance(path_change, Deleted):
                if rel_path in new_path2module:
                    mod = new_path2module.pop(rel_path)
                    changed[mod.mname] = TsModuleChange.from_modules(Deleted(mod))
            elif isinstance(path_change, Modified):
                if rel_path in new_path2module:
                    mod_new, _ = parse_typescript_module_script(project, rel_path)
                    mod_old = path2module[rel_path]
                    new_path2module[rel_path] = mod_new
                    changed[mod_new.mname] = TsModuleChange.from_modules(Modified(mod_old, mod_new))

        # --- 集成 post_edit_analysis ---
        post_analysis = None
        if hasattr(change_processor, 'post_edit_analysis'):
            post_analysis = change_processor.post_edit_analysis(
                new_path2module, changed
            )
        else:
            post_analysis = {}

        # 切回当前 commit，做 pre_edit_analysis
        checkout_commit(commit_now.hash)
        pre_analysis = None
        if hasattr(change_processor, 'pre_edit_analysis'):
            pre_analysis = change_processor.pre_edit_analysis(
                path2module, changed
            )
        else:
            pre_analysis = {}
        # 切回下一个 commit，准备 process_change
        checkout_commit(commit_next.hash)

        processed = change_processor.process_change(
            changed,
            pre_analysis,
            post_analysis,
        )
        results.extend(processed)
        commit_now = commit_next
        path2module = new_path2module

    return results 


def edits_from_ts_commit_history(
    project: Path,
    history: Sequence[CommitInfo],
    tempdir: Path,
    change_processor,
    silent: bool = False,
    time_limit: float | None = None,
) -> Sequence:
    """
    从 TypeScript 仓库的 git 历史中提取编辑信息。
    这是 _edits_from_ts_commit_history 的公共包装函数。
    """
    # 创建临时工作目录
    workdir = tempdir / "ts_edits"
    workdir.mkdir(parents=True, exist_ok=True)
    
    # 复制 .git 目录到工作目录
    git_dir = workdir / ".git"
    if not git_dir.exists():
        import shutil
        shutil.copytree(project / ".git", git_dir)
    
    # 设置忽略目录（暂时为空）
    ignore_dirs = set()
    
    try:
        return _edits_from_ts_commit_history(
            workdir,
            history,
            change_processor,
            ignore_dirs,
            silent,
            time_limit,
        )
    finally:
        # 清理临时目录
        if workdir.exists():
            import shutil
            shutil.rmtree(workdir) 