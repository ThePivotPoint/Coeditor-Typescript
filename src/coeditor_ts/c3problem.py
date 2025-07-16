from dataclasses import replace
from pprint import pprint
from textwrap import indent


from ._utils import scalar_stats

from .change import Added, Change, Modified, show_change
from .common import *
from .encoding import (
    Add_id,
    Del_id,
    DeltaKey,
    N_Extra_Ids,
    Newline_id,
    TkDelta,
    TokenizedEdit,
    TruncateAt,
    break_into_chunks,
    change_tks_to_original_delta,
    change_to_line_diffs,
    change_to_tokens,
    decode_tokens,
    encode_lines_join,
    encode_single_line,
    get_extra_id,
    line_diffs_to_original_delta,
    tk_splitlines,
    tokens_to_change,
    truncate_output_tks,
    truncate_section,
    truncate_sections,
)
from .git import CommitInfo
from .scoped_changes import (
    TsChangedSpan,
    TsChangeScope,
    TsModule,
    TsModuleChange,
    TsProjectChange,
    LineRange,
    ModuleName,
    ProjectChangeProcessor,
    ProjectPath,
    TsProjectState,
    TsStatementSpan,
)
from .tk_array import TkArray


@dataclass(frozen=True)
class ChangedHeader:
    """Represents the changes made to a header.
    This format does not store parent syntax nodes and is more suitable for serialization.
    """

    change_tks: TkArray
    # below are pre-edit attributes
    type: str
    line_range: LineRange
    path: ProjectPath

    def __repr__(self) -> str:
        return (
            f"ChangedHeader(path={self.path}, range={self.line_range}, "
            f"change={self.change_tks})"
        )


@dataclass(frozen=True)
class ChangedCodeSpan:
    """Represents the changes made to a span of code.
    This format does not store parent syntax nodes and is more suitable for serialization.
    """

    headers: Sequence[ChangedHeader]
    original: TkArray
    delta: TkDelta
    # below are pre-edit attributes
    line_range: LineRange
    module: ModuleName

    def get_change(self) -> Modified[str]:
        change_tks = self.delta.apply_to_change(self.original.tolist())
        return tokens_to_change(change_tks)

    def change_size(self) -> int:
        return len(self.original) + self.delta.change_size()


class SrcInfo(TypedDict):
    project: str
    commit: CommitInfo | None


@dataclass(frozen=True)
class TsC3Problem:
    "Contextual code change prediction problem."
    span: ChangedCodeSpan
    # The line ids in the change tks that should be edited
    edit_line_ids: Sequence[int]
    # most relevant to least relevant
    relevant_changes: Sequence[ChangedCodeSpan]
    # most relevant to least relevant
    relevant_unchanged: Mapping["PyFullName", "TsDefinition"]
    # some optional information about how the problem was generated
    change_type: Change[None]
    src_info: SrcInfo
    transformations: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.edit_line_ids:
            raise ValueError(f"edit_line_ids is empty. Problem: {self.summary()}")

    def restrict_span_changes(self):
        "restrict the changes in the span to the edit lines"
        eids = self.edit_line_ids
        delta = self.span.delta.for_input_range((eids[0], eids[-1] + 1))
        span = replace(self.span, delta=delta)
        return replace(self, span=span)

    @property
    def path(self) -> ProjectPath:
        return self.span.headers[-1].path

    def uid(self) -> tuple[ProjectPath, str]:
        return self.path, not_none(self.src_info["commit"]).hash

    def meta_data_lines(self) -> list[str]:
        return [
            f"path: {self.span.headers[-1].path}",
            f"project: {self.src_info['project']}",
            f"commit: {self.src_info['commit']}",
        ]

    def summary(self) -> str:
        return "\n".join(self.meta_data_lines())

    def show(self) -> str:
        return show_sections(
            ("summary", self.summary()),
            ("delta", str(self.span.delta)),
            ("main input", decode_tokens(self.span.original.tolist())),
            ("edit_line_ids", str(self.edit_line_ids)),
        )

    def print(self):
        print(self.show())

    def line_ids_to_input_lines(self, line_ids: Sequence[int]) -> Sequence[int]:
        """Convert the edit lines (which are line ids including deleted lines) into
        normal line numbers that do not include deleted lines."""
        change_tks = self.span.delta.apply_to_change(self.span.original.tolist())
        input_l = self.span.line_range[0]
        input_lines = list[int]()
        for i, tks in enumerate(tk_splitlines(change_tks)):
            if tks and tks[0] == Del_id:
                continue
            if i in line_ids:
                input_lines.append(input_l)
            input_l += 1

        return input_lines


PyFullName = NewType("PyFullName", str)


@dataclass
class TsDefinition:
    """
    TypeScript 符号定义信息，支持函数、类、方法、变量等。
    - full_name: 全限定名（如 module.Class.method）
    - start_locs: 所有定义出现的起始位置（行号, 列号）
    - signatures: 所有签名（如函数签名、类定义等）
    """
    full_name: str
    start_locs: set[tuple[int, int]]
    signatures: set[str]

    def __post_init__(self):
        self.parent = ".".join(split_dots(self.full_name)[:-1])

    def update(self, node, source_code: str):
        """
        node: tree_sitter.Node
        source_code: 该文件源码字符串
        """
        # 支持的 TypeScript 节点类型
        valid_types = {"function_declaration", "class_declaration", "method_definition", "variable_declaration"}
        if node.type not in valid_types:
            return
        # 记录起始位置
        self.start_locs.add(node.start_point)
        # 提取签名
        sig = self._extract_signature(node, source_code)
        if sig:
            self.signatures.add(sig)

    def _extract_signature(self, node, source_code: str) -> str:
        """
        提取 TypeScript 节点的签名字符串
        """
        # 直接用节点的源码片段作为签名（可根据需要自定义更复杂的提取）
        start_byte = node.start_byte
        end_byte = node.end_byte
        return source_code[start_byte:end_byte].strip()


@dataclass(frozen=True)
class LineUsageAnalysis:
    line2usages: Mapping[int, Sequence[TsDefinition]]

    def __repr__(self):
        lines = (
            ["LineUsageAnalysis("]
            + [f"    {l}: {us}" for l, us in self.line2usages.items()]
            + [")"]
        )
        return "\n".join(lines)


@dataclass
class JediUsageAnalyzer:
    include_parent_usages: bool = True
    include_builtins: bool = False

    _KnownJediErrors = {
        "not enough values to unpack (expected 2",
        "'Newline' object has no attribute 'children'",
        "trailer_op is actually ",
        "There's a scope that was not managed: <Module",
        "maximum recursion depth exceeded",
        "'NoneType' object has no attribute 'type'",
    }

    def __post_init__(self):
        self.error_counts = dict[str, int]()
        self.tlogger: TimeLogger = TimeLogger()

    def get_line_usages(
        self,
        ts_module,  # TsParsedModule
        lines_to_analyze: Collection[int],
        silent: bool = False,
    ):
        """
        使用 tree-sitter 分析 TypeScript 代码，收集每一行的符号定义和用法。
        """
        from tree_sitter import Node
        
        code = ts_module.code
        tree = ts_module.tree
        line2usages = dict[int, set[PyFullName]]()
        name2def_node = dict[PyFullName, list[Node]]()
        registered_classes = set[PyFullName]()

        def register_usage(node: Node, usages: set[PyFullName], full_name: str):
            """注册符号用法"""
            if not full_name:
                return
            if not self.include_builtins and full_name.startswith("builtins."):
                return
            fname = PyFullName(full_name)
            usages.add(fname)
            name2def_node.setdefault(fname, list()).append(node)

        def register_class_usage(node: Node, usages: set[PyFullName], class_name: str):
            """注册类用法"""
            if not class_name or class_name in registered_classes:
                return
            if not self.include_parent_usages and class_name.startswith("builtins."):
                return
            # 对于类，我们注册类本身和其成员
            register_usage(node, usages, class_name)
            registered_classes.add(PyFullName(class_name))

        def extract_identifier(node: Node) -> str:
            """提取节点的标识符名称"""
            for child in node.children:
                if child.type == "identifier":
                    return code[child.start_byte:child.end_byte]
            return ""

        def visit_node(node: Node, parent_name: str = ""):
            """递归访问 AST 节点"""
            node_type = node.type
            
            # 处理函数声明
            if node_type == "function_declaration":
                name = extract_identifier(node)
                if name:
                    full_name = parent_name + "." + name if parent_name else name
                    line = node.start_point[0] + 1  # tree-sitter 行号从0开始
                    if line in lines_to_analyze:
                        usages = line2usages.setdefault(line, set())
                        register_usage(node, usages, full_name)
                    # 递归处理函数体
                    for child in node.children:
                        visit_node(child, full_name)
                        
            # 处理类声明
            elif node_type == "class_declaration":
                name = extract_identifier(node)
                if name:
                    full_name = parent_name + "." + name if parent_name else name
                    line = node.start_point[0] + 1
                    if line in lines_to_analyze:
                        usages = line2usages.setdefault(line, set())
                        register_class_usage(node, usages, full_name)
                    # 递归处理类体
                    for child in node.children:
                        visit_node(child, full_name)
                        
            # 处理方法定义
            elif node_type == "method_definition":
                name = extract_identifier(node)
                if name:
                    full_name = parent_name + "." + name if parent_name else name
                    line = node.start_point[0] + 1
                    if line in lines_to_analyze:
                        usages = line2usages.setdefault(line, set())
                        register_usage(node, usages, full_name)
                        
            # 处理变量声明
            elif node_type == "variable_declaration":
                for child in node.children:
                    if child.type == "variable_declarator":
                        name = extract_identifier(child)
                        if name:
                            full_name = parent_name + "." + name if parent_name else name
                            line = node.start_point[0] + 1
                            if line in lines_to_analyze:
                                usages = line2usages.setdefault(line, set())
                                register_usage(node, usages, full_name)
                        break
                        
            # 处理其他节点类型
            else:
                for child in node.children:
                    visit_node(child, parent_name)

        try:
            # 开始遍历 AST
            visit_node(tree.root_node)
            
            # 构建 TsDefinition 对象
            name2pydef = dict[PyFullName, TsDefinition]()
            for fname, nodes in name2def_node.items():
                pdef = TsDefinition(fname, set(), set())
                for node in nodes:
                    pdef.update(node, source_code=code)
                name2pydef[fname] = pdef
                
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            err_text = repr(e)
            str_limit = 80
            if len(err_text) > str_limit:
                err_text = err_text[:str_limit] + "..."
            self.add_error(err_text)

        return LineUsageAnalysis(
            {
                k: [name2pydef[n] for n in sorted(names) if n in name2pydef]
                for k, names in line2usages.items()
            }
        )

    def add_error(self, err_text: str):
        self.error_counts[err_text] = self.error_counts.get(err_text, 0) + 1

    @staticmethod
    def is_known_error(err_text: str):
        return any(k in err_text for k in JediUsageAnalyzer._KnownJediErrors)


@dataclass
class _C3PreAnalysis:
    """
    - `training_samples`: The set of spans that should be used as training examples.
    This might include some unchanged spans if `sample_unmodified_ratio > 0`.
    - `usage_analysis`: Map each module to its line usage analysis. Only lines
    that will be used later are analyzed.
    """

    training_samples: Set[tuple[ModuleName, LineRange]]
    usage_analysis: Mapping[ModuleName, LineUsageAnalysis]


@dataclass
class TsC3ProblemGenerator(ProjectChangeProcessor[TsC3Problem]):
    """
    Generate `TsC3Problem`s from git histories.

    ### Change log
    - v3.1: fix self-references in static analaysis
    - v3.0: Support `sample_unmodified_ratio`. Fix missing changes in `relevant_changes`.
    - v2.9: Add sibling usages for class members. Improve statement signatures.
        - fix 1: Remove builtin usages by default.
    - v2.8: Fix module usages in `pre_edit_analysis`. Sort changes using heuristic.
    - v2.7: Use new PyDefiniton that includes signatures.
    - v2.6: fix missing changes in `JModuleChanges`. Rename to edit_line_ids.
    - v2.5: fix newline encoding bug.
    - v2.4: fix buggy encoding of `Added` and `Deleted` changes.
    - v2.3: always generate problems with full editing range and move the problem
    splitting logic elsewhere. Also changed the data format of `ChangedCodeSpan`.
    """

    VERSION = "3.1"
    # change spans with more than this many lines will be ignored
    max_span_lines: int = 500
    # change spans with more than this many characters will be ignored
    max_span_chars: int = 6000
    # the ratio of unmodified spans to be sampled
    neg_to_pos_ratio: float = 0.0
    analyzer: JediUsageAnalyzer = field(default_factory=JediUsageAnalyzer)

    def __repr__(self) -> str:
        return repr_modified_args(self)

    def append_stats(self, stats: dict[str, Any]) -> None:
        rec_add_dict_to(stats, {"analyzer_errors": self.analyzer.error_counts})

    def clear_stats(self) -> None:
        return self.analyzer.error_counts.clear()

    def use_unchanged(self) -> bool:
        return self.neg_to_pos_ratio > 0

    # 【guohx】预编辑分析函数，在代码变更发生前进行分析，为后续的问题生成做准备
    def pre_edit_analysis(
        self,
        pstate: TsProjectState,  # 【guohx】项目状态对象，包含TypeScript模块信息
        modules: Mapping[RelPath, TsModule],  # 【guohx】模块映射，键为相对路径，值为TsModule对象
        changes: Mapping[ModuleName, TsModuleChange],  # 【guohx】模块变更映射，键为模块名，值为模块变更对象
    ) -> _C3PreAnalysis:  # 【guohx】返回预分析结果，包含训练样本和用法分析
        # 【guohx】首先，采样未修改的代码片段作为负样本
        selected_set = set[tuple[ModuleName, LineRange]]()  # 【guohx】选中的训练样本集合（模块名，行范围）
        negative_set = set[tuple[ModuleName, LineRange]]()  # 【guohx】负样本集合（模块名，行范围）
        # 【guohx】遍历所有模块变更
        for mname, mchange in changes.items():
            # 【guohx】遍历每个变更中的代码片段
            for cspan in mchange.changed:
                # 【guohx】检查是否应该为这个代码片段生成问题
                if not self.should_mk_problem(
                    cspan,  # 【guohx】代码片段
                    func_only=not self.is_training,  # 【guohx】是否只处理函数（非训练模式下）
                    max_chars=self.max_span_chars,  # 【guohx】最大字符数限制
                    max_lines=self.max_span_lines,  # 【guohx】最大行数限制
                ):
                    continue  # 【guohx】如果不满足条件，跳过
                # 【guohx】如果是修改类型的变更
                if isinstance(cspan.change, Modified):
                    if cspan.change.unchanged:  # 【guohx】如果包含未修改的部分
                        negative_set.add((mname, cspan.line_range))  # 【guohx】添加到负样本集合
                    else:  # 【guohx】如果包含修改的部分
                        selected_set.add((mname, cspan.line_range))  # 【guohx】添加到正样本集合

        # 【guohx】包含一些负样本作为训练数据，用于平衡正负样本比例
        if negative_set:  # 【guohx】如果存在负样本
            # 【guohx】计算选择概率：正样本数量 * 负样本比例 / 负样本总数
            select_prob = len(selected_set) * self.neg_to_pos_ratio / len(negative_set)
            # 【guohx】随机选择负样本
            for x in negative_set:
                if random.random() < select_prob:  # 【guohx】根据概率决定是否选择
                    selected_set.add(x)  # 【guohx】添加到训练样本集合

        # 【guohx】初始化用法分析字典
        usages = dict[ModuleName, LineUsageAnalysis]()  # 【guohx】模块名到行用法分析的映射
        # 【guohx】构建模块名到文件路径的映射
        src_map = {str(m.mname): f for f, m in modules.items()}  # 【guohx】模块名 -> 相对路径，转换为字符串
        # 【guohx】遍历每个模块变更
        for mname, mchange in changes.items():
            # 【guohx】初始化该模块的用法分析
            usages[mname] = LineUsageAnalysis({})  # 【guohx】空的行用法分析
            # 【guohx】初始化需要分析的行号集合
            lines_to_analyze = set[int]()

            # 【guohx】遍历该模块中的所有变更片段
            for span in mchange.changed:
                # 【guohx】如果这个片段不在选中的训练样本中，跳过分析
                if (mname, span.line_range) not in selected_set:
                    continue  # 【guohx】跳过分析
                # 【guohx】添加代码片段的行范围到分析集合
                lines_to_analyze.update(span.line_range.to_range())  # 【guohx】添加主要行范围
                # 【guohx】添加头部行范围到分析集合
                lines_to_analyze.update(span.header_line_range.to_range())  # 【guohx】添加头部行范围
            # 【guohx】如果没有需要分析的行，跳过
            if not lines_to_analyze:
                continue

            # 【guohx】获取模块对应的文件路径
            mod_path = src_map.get(str(mname))  # 【guohx】从映射中获取相对路径，转换为字符串
            if mod_path is None:
                continue  # 【guohx】如果找不到对应的路径，跳过
            # 【guohx】获取对应的TypeScript模块对象
            ts_module = pstate.modules.get(str(mod_path))  # 【guohx】从项目状态中获取模块，转换为字符串
            if ts_module is None:
                continue  # 【guohx】如果找不到对应的模块，跳过
            # 【guohx】使用分析器获取行用法分析
            line_usages = self.analyzer.get_line_usages(  # 【guohx】调用TypeScript用法分析器
                ts_module,  # 【guohx】TypeScript模块对象
                lines_to_analyze,  # 【guohx】需要分析的行号集合
                silent=True  # 【guohx】静默模式
            )
            # 【guohx】将分析结果存储到用法字典中
            usages[mname] = line_usages  # 【guohx】保存该模块的用法分析结果
        # 【guohx】返回预分析结果对象
        return _C3PreAnalysis(
            training_samples=selected_set,  # 【guohx】训练样本集合
            usage_analysis=usages,  # 【guohx】用法分析结果
        )

    # 【guohx】后编辑分析函数，在代码变更发生后进行分析，主要用于确定模块的拓扑排序
    def post_edit_analysis(
        self,
        pstate: TsProjectState,  # 【guohx】项目状态对象，包含Jedi项目和脚本信息
        modules: Mapping[RelPath, TsModule],  # 【guohx】模块映射，键为相对路径，值为JModule对象
        changes: Mapping[ModuleName, TsModuleChange],  # 【guohx】模块变更映射，键为模块名，值为模块变更对象
    ) -> list[ModuleName]:  # 【guohx】返回模块的拓扑排序列表
        "Return the topological order among the modules."
        # 【guohx】对模块进行拓扑排序
        # 【guohx】初始化模块依赖字典，键为模块名，值为依赖的模块集合
        module_deps = dict[ModuleName, set[ModuleName]]()  # 【guohx】模块依赖关系映射
        # 【guohx】遍历所有模块
        for rel_path, module in modules.items():
            # 【guohx】获取该模块导入的所有名称
            names = {n for n in module.imported_names}  # 【guohx】该模块导入的名称集合
            # 【guohx】获取对应的TypeScript模块对象
            ts_module = pstate.modules.get(str(rel_path))  # 【guohx】从项目状态中获取模块，转换为字符串
            # 【guohx】获取或创建该模块的依赖集合
            deps = module_deps.setdefault(module.mname, set())  # 【guohx】该模块的依赖模块集合
            # 【guohx】遍历每个导入的名称
            for n in names:
                try:
                    # 【guohx】使用tree-sitter分析导入的模块
                    # 这里简化处理，直接添加导入的模块名作为依赖
                    # 在实际实现中，需要更复杂的模块解析逻辑
                    if n in module.imported_names:
                        # 假设导入的名称对应模块名
                        deps.add(n)
                except Exception as e:
                    # 【guohx】如果查找过程中出现异常，记录错误并继续
                    self.analyzer.add_error(str(e))  # 【guohx】将错误添加到分析器
                    continue  # 【guohx】跳过这个名称
        # 【guohx】根据依赖关系对模块进行拓扑排序
        module_order = self._topological_sort(module_deps)  # 【guohx】调用拓扑排序函数
        # 【guohx】返回排序后的模块列表
        return module_order

    def _topological_sort(self, module_deps: dict[ModuleName, set[ModuleName]]) -> list[ModuleName]:
        """简单的拓扑排序实现"""
        # 计算入度
        in_degree = {module: 0 for module in module_deps}
        for deps in module_deps.values():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        # 找到入度为0的节点
        queue = [module for module, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # 更新依赖当前模块的其他模块的入度
            for module, deps in module_deps.items():
                if current in deps:
                    in_degree[module] -= 1
                    if in_degree[module] == 0:
                        queue.append(module)
        
        # 如果还有节点未处理，说明有环，返回所有模块
        if len(result) < len(module_deps):
            return list(module_deps.keys())
        
        return result

    def process_change(
        self,
        pchange: TsProjectChange,
        pre_analysis: _C3PreAnalysis,
        module_order: Sequence[ModuleName],
    ) -> Sequence[TsC3Problem]:
        """
        Return (untransformed) c3 problems from the given project change.
        Each problem contains a code change, a list of relevant (previous) changes,
        and other extra informaton about the code change.
        """
        before_mod_map = {str(m.mname): m for m in pchange.all_modules.before}
        cache = C3GeneratorCache(before_mod_map)
        src_info: SrcInfo = {
            "project": pchange.project_name,
            "commit": pchange.commit_info,
        }

        prev_cspans = list[ChangedCodeSpan]()
        problems = list[TsC3Problem]()
        mod2usages = pre_analysis.usage_analysis
        for m in module_order:
            if (mchange := pchange.changed.get(m)) is None:
                continue
            if not (usages := mod2usages.get(m)):
                usages = LineUsageAnalysis({})
                warnings.warn("Unexpected: usages missing for module: " + str(m))
            for span in mchange.changed:
                if (m, span.line_range) in pre_analysis.training_samples:
                    code_span = cache.to_code_span(span)
                    # latest changes are more relevant
                    relevant_unchanged = cache.get_relevant_unchanged(code_span, usages)
                    relevant_changes = cache.sort_changes(
                        code_span, relevant_unchanged, reversed(prev_cspans)
                    )

                    n_lines = span.line_range[1] - span.line_range[0]
                    prob = TsC3Problem(
                        code_span,
                        range(0, n_lines + 1),  # one additional line for appending
                        relevant_changes=relevant_changes,
                        relevant_unchanged=relevant_unchanged,
                        change_type=span.change.map(lambda _: None),
                        src_info=src_info,
                    )
                    problems.append(prob)
                    if code_span.delta:
                        prev_cspans.append(code_span)
                else:
                    if span.change.changed:
                        prev_cspans.append(cache.to_code_span(span))
        return problems


class C3GeneratorCache:
    """
    Cache various information needed for constructing C3 problems.
    A new cache should be created for each project change (i.e., each commit).
    """

    def __init__(self, pre_module_map: Mapping[str, TsModule]):
        # stores the changed headers
        self._header_cache = dict[str, ChangedHeader]()
        # stores the definitions pre-edit
        self._pre_def_cache = dict[str, list[ChangedCodeSpan]]()
        # stores the changed code spans
        self._cspan_cache = dict[tuple[str, LineRange], ChangedCodeSpan]()
        self._module_map = pre_module_map
        # 暂时注释掉 ModuleHierarchy，因为未定义
        # self._mod_hier = ModuleHierarchy.from_modules(pre_module_map)

    def set_module_map(self, pre_module_map: Mapping[str, TsModule]):
        self._module_map = pre_module_map
        # 暂时注释掉 ModuleHierarchy，因为未定义
        # self._mod_hier = ModuleHierarchy.from_modules(pre_module_map)

    def clear_caches(
        self, pre_changed: set[str], post_changed: set[str]
    ) -> None:
        "Clear outdated caches."
        for k in tuple(self._header_cache):
            if k in pre_changed or k in post_changed:
                del self._header_cache[k]
        for k in tuple(self._pre_def_cache):
            if k in pre_changed:
                del self._pre_def_cache[k]
        for k in tuple(self._cspan_cache):
            if k[0] in pre_changed or k[0] in post_changed:
                del self._cspan_cache[k]

    def create_problem(
        self,
        target: TsChangedSpan,
        target_lines: Sequence[int],
        changed: Mapping[ModuleName, TsModuleChange],
        target_usages: LineUsageAnalysis,
        src_info: SrcInfo,
    ) -> TsC3Problem:
        relevant_changes = list[ChangedCodeSpan]()
        changed = dict(changed)
        module = target.module
        target_mc = changed.pop(module)
        all_mc = [target_mc] + list(changed.values())
        for mc in all_mc:
            is_target_mc = mc.module_change.earlier.mname == module
            for cspan in mc.changed:
                if not is_target_mc or cspan.line_range != target.line_range:
                    relevant_changes.append(self.to_code_span(cspan))

        code_span = self.to_code_span(target)
        changed_code = code_span.delta.apply_to_change(code_span.original.tolist())
        target_set = set(target_lines)
        line_ids = list[int]()
        input_l = target.line_range[0]
        for i, tks in enumerate(tk_splitlines(changed_code)):
            if tks and tks[0] == Del_id:
                continue
            if input_l in target_set:
                line_ids.append(i)
            input_l += 1
        code_span = replace(
            code_span, original=TkArray.new(changed_code), delta=TkDelta.empty()
        )
        relevant_unchanged = self.get_relevant_unchanged(code_span, target_usages)

        relevant_changes = self.sort_changes(
            code_span, relevant_unchanged, relevant_changes
        )

        prob = TsC3Problem(
            code_span,
            line_ids,
            relevant_changes=relevant_changes,
            relevant_unchanged=relevant_unchanged,
            change_type=target.change.map(lambda _: None),
            src_info=src_info,
        )
        return prob

    def get_relevant_unchanged(
        self,
        this_change: ChangedCodeSpan,
        line_usages: LineUsageAnalysis,
    ):
        module = this_change.module
        # parent defs are also considered as used
        name2def = dict[PyFullName, TsDefinition]()
        all_lines = set(this_change.line_range.to_range())
        all_lines.update(this_change.headers[-1].line_range.to_range())
        for l in all_lines:
            for pydef in line_usages.line2usages.get(l, []):
                if pydef.full_name.startswith(module) and any(
                    l in all_lines for l, _ in pydef.start_locs
                ):
                    # skip self references
                    continue
                name2def.setdefault(PyFullName(pydef.full_name), pydef)
        return {k: name2def[k] for k in sorted(name2def.keys())}

    max_distance_penalty = 1000
    usage_bonus = 2000

    def sort_changes(
        self,
        target: ChangedCodeSpan,
        used_defs: Mapping[PyFullName, TsDefinition],
        changed: Iterable[ChangedCodeSpan],
    ) -> Sequence[ChangedCodeSpan]:
        def distance_penalty(cspan: ChangedCodeSpan) -> int:
            if cspan.module != target.module:
                return self.max_distance_penalty
            dis_above = abs(target.line_range[0] - cspan.line_range[1])
            dis_below = abs(cspan.line_range[0] - target.line_range[1])
            return min(self.max_distance_penalty, dis_above, dis_below)

        def usage_penalty(cspan: ChangedCodeSpan) -> int:
            path = cspan.headers[-1].path
            fullname = path.module + "." + path.path
            if fullname in used_defs:
                return -self.usage_bonus
            return 0

        def length_penalty(cspan: ChangedCodeSpan) -> int:
            return len(cspan.original) + cspan.delta.change_size()

        result = list(changed)
        result.sort(
            key=lambda x: distance_penalty(x) + usage_penalty(x) + length_penalty(x)
        )
        return result

    def to_header(self, cs: Change[TsChangeScope]) -> ChangedHeader:
        path = str(cs.earlier.path)  # 转换为字符串
        if (ch := self._header_cache.get(path)) is None:
            header_change = cs.map(lambda s: s.header_code.strip("\n"))
            ch = ChangedHeader(
                TkArray.new(change_to_tokens(header_change)),
                cs.earlier.tree.type,
                cs.earlier.header_line_range,
                cs.earlier.path,
            )
            self._header_cache[path] = ch
        return ch

    def to_code_span(self, span: TsChangedSpan):
        # 简化处理，直接使用模块名
        mod = str(span.module) if hasattr(span, 'module') else "unknown"
        key = (mod, span.line_range)
        if (cs := self._cspan_cache.get(key)) is not None:
            return cs

        original, delta = change_tks_to_original_delta(change_to_tokens(span.change))
        result = ChangedCodeSpan(
            headers=[self.to_header(cs) for cs in span.parent_scopes],
            original=TkArray.new(original),
            delta=delta,
            line_range=span.line_range,
            module=span.module,
        )
        self._cspan_cache[key] = result
        return result


class TsC3ProblemTransform(ABC):
    "A strategy to generate new C3 problems from the orginal ones."

    @abstractmethod
    def transform(self, prob: TsC3Problem) -> Sequence[TsC3Problem]:
        ...


@dataclass
class C3ProblemSimpleSplit(TsC3ProblemTransform):
    "Simply split the problem into fixed-sized editing ranges."
    VERSION = "1.1"

    max_lines_to_edit: int = 30
    max_split_factor: int = 4
    allow_empty_problems: bool = True

    def transform(self, prob: TsC3Problem) -> Sequence[TsC3Problem]:
        delta = prob.span.delta
        l_range = prob.edit_line_ids
        assert isinstance(l_range, range)
        start, stop = l_range.start, l_range.stop
        problems = list[TsC3Problem]()
        new_trans = prob.transformations + ("split",)
        for i in range(start, stop, self.max_lines_to_edit):
            j = min(i + self.max_lines_to_edit, stop)
            sub_delta = delta.for_input_range((i, j))
            if sub_delta.num_changes() > 0 or (self.allow_empty_problems and i < j):
                sub_prob = replace(
                    prob, edit_line_ids=range(i, j), transformations=new_trans
                )
                problems.append(sub_prob)
            if len(problems) >= self.max_split_factor:
                break
        return problems


@dataclass
class TsC3ProblemChangeInlining(TsC3ProblemTransform):
    """Split the problem into fixed-sized editing ranges like `C3ProblemSimpleSplit`,
    but also randomly keep some subset of changes in the input.

    ### Change log
    - v1.4: add `allow_empty_problems` option. Improve inlining sampling strategy.
    - v1.3: make `random_subset` truely random.
    - v1.2: fix newline encoding bug.
    - v1.1
        - Dropout changes using change groups instead of individual change actions.
        - Perform dropout at entire problem level ratehr than chunk level. This way,
    changes in later chunks will be visible as well.
        - Removed `dropout_prob`.
    """

    VERSION = "1.4"

    max_lines_to_edit: int = 30
    max_split_factor: int = 4
    # when dropping the changes into the input, the biggest ratio of changes to inline
    max_inline_ratio: float = 1.0
    _test_prob: float = 0.01
    allow_empty_problems: bool = True

    def __post_init__(self):
        self._rng = random.Random()

    def transform(self, prob: TsC3Problem) -> Sequence[TsC3Problem]:
        original = prob.span.original
        delta = prob.span.delta
        l_range = prob.edit_line_ids
        assert isinstance(l_range, range)
        start, stop = l_range.start, l_range.stop

        grouped_keys = delta.change_groups()
        if len(grouped_keys) >= 2:
            keys_to_inline = list[DeltaKey]()
            # bias toward smaller ratio
            ratio = self.max_inline_ratio * random.random() ** 2
            for group in grouped_keys:
                if random.random() <= ratio:
                    keys_to_inline.extend(group)
        else:
            keys_to_inline = []
        if keys_to_inline:
            delta1, delta2 = delta.decompose_for_change(keys_to_inline)
            if random.random() < self._test_prob:
                result1 = delta2.apply_to_change(
                    delta1.apply_to_change(original.tolist())
                )
                result2 = delta.apply_to_change(original.tolist())
                code1 = tokens_to_change(result1).after
                code2 = tokens_to_change(result2).after
                if code1 != code2:
                    print_sections(
                        ("result1", decode_tokens(result1)),
                        ("result2", decode_tokens(result2)),
                        ("delta", str(delta)),
                        ("keys_to_drop", str(keys_to_inline)),
                        ("delta1", str(delta1)),
                        ("delta2", str(delta2)),
                    )
                    raise AssertionError("decompose_for_change failed.")
            delta2_groups = delta2.change_groups()
            if not self.allow_empty_problems and not delta2_groups:
                print_err(f"{delta=}, {keys_to_inline=}, {delta1=}")
                raise AssertionError("Empty delta2_groups")
            new_original = TkArray.new(delta1.apply_to_change(original.tolist()))
            new_trans = prob.transformations + ("split", "dropout")
            new_span = replace(prob.span, original=new_original, delta=delta2)
        else:
            new_trans = prob.transformations + ("split",)
            new_span = prob.span
            delta1 = None
            delta2_groups = delta.change_groups()

        prob_count = list[tuple[TsC3Problem, int]]()
        for i in range(start, stop, self.max_lines_to_edit):
            j = min(i + self.max_lines_to_edit, stop)
            edit_line_ids = range(i, j)
            if delta1 is not None:
                edit_line_ids = delta1.get_new_line_ids(edit_line_ids)
            line_set = set(edit_line_ids)
            n_groups = sum(any(key[0] in line_set for key in g) for g in delta2_groups)
            if n_groups > 0 or (self.allow_empty_problems and edit_line_ids):
                sub_prob = replace(
                    prob,
                    span=new_span,
                    edit_line_ids=edit_line_ids,
                    transformations=new_trans,
                )
                prob_count.append((sub_prob, n_groups))
        # return the problems with the most changes
        prob_count.sort(key=lambda p: p[1], reverse=True)
        probs = [p[0] for p in prob_count]
        if self.allow_empty_problems and not probs:
            raise AssertionError(f"No problems generated for:\n{prob.show()}")
        return probs[: self.max_split_factor]


CompletionKind = Literal["add", "mod"]


@dataclass
class C3ToCodeCompletion(TsC3ProblemTransform):
    """Convert the C3 problem into an edit-oriented code completion problem by
    randomly picking a changed line as the completion target, deleting its
    old version, and treating the new version as the desired output.

    ### Change log
    - v1.2: add `use_modifications`. change `addition_only` to `use_additions`.
    - v1.1: add `addition_only`.
    """

    VERSION = "1.2"
    min_target_size: int = 6
    use_additions: bool = True
    use_modifications: bool = True

    def __post_init__(self):
        if not (self.use_additions or self.use_modifications):
            warnings.warn("Both use_additions and use_modifications are False.")

    def extract_completion(
        self, original: TokenSeq, delta: TkDelta
    ) -> tuple[TokenSeq, TkDelta, CompletionKind] | None:
        """
        Try to extract a code completion instance from the given change, return None if
        not suitable. This works by taking the last addition from the changes as the
        code completion target and applying all the changes before it to the original
        code to get the context. Note that if this addition is part of a replacement,
        the deletion is applied to the context as well.
        """

        target = delta.change_groups()[-1]
        acts = [delta[k] for k in target]
        good = (
            len(acts) <= 2
            and acts[0][0] == Add_id
            and (
                (self.use_additions and acts[-1][0] == Add_id)
                or (self.use_modifications and acts[-1][0] == Del_id)
            )
            and len(acts[0]) >= self.min_target_size
        )
        if not good:
            return None

        kind = "add" if acts[-1][0] == Add_id else "mod"
        prev_changes = [k for k in delta.keys() if k < target[0]]
        if acts[-1][0] == Del_id:
            # if the last change is a deletion, move it into prev_changesine into before_changes
            prev_changes.append(target[-1])
            target = target[:-1]
            assert target

        prev_delta, rest_delta = delta.decompose_for_change(prev_changes)
        new_original = prev_delta.apply_to_change(original)
        new_delta_keys = tuple(rest_delta.keys())[: len(target)]
        new_delta = rest_delta.for_keys(new_delta_keys)
        assert new_delta, "the remaining delta should not be empty"
        return new_original, new_delta, kind

    def transform(self, prob: TsC3Problem) -> Sequence[TsC3Problem]:
        original = prob.span.original.tolist()
        delta = prob.span.delta

        sampled = self.extract_completion(original, delta)
        if sampled is None:
            return []
        original, delta, kind = sampled
        new_span = replace(prob.span, original=TkArray.new(original), delta=delta)
        new_trans = prob.transformations + ("code_completion", kind)
        new_lines = tuple(set(k[0] for k in delta.keys()))
        new_prob = replace(
            prob,
            span=new_span,
            edit_line_ids=new_lines,
            transformations=new_trans,
        )
        return [new_prob]


@dataclass(frozen=True)
class TkC3Problem(TokenizedEdit):
    "Tokenized contextual code change prediction problem."
    main_input: TkArray
    header: TkArray
    output: TkArray
    path: ProjectPath
    change_type: Change[None]
    # most relevant to least relevant
    named_references: Sequence[tuple[str, TkArray]]
    project: str
    commit: CommitInfo | None
    truncated: bool

    @property
    def references(self) -> Sequence[TkArray]:
        return [ref for _, ref in self.named_references]

    def __repr__(self):
        return f"TkC3Problem(path={self.path}, type={self.change_type.as_char()}, stats={self.stats()})"

    @property
    def input_tks(self) -> TokenSeq:
        return self.header.tolist() + self.main_input.tolist()

    @property
    def output_tks(self) -> TokenSeq:
        return self.output.tolist()

    @property
    def main_tks(self) -> TokenSeq:
        return self.input_tks

    def all_ctxs(self) -> dict[str, TokenSeq]:
        return {name: ref.tolist() for name, ref in self.named_references}

    def meta_data_lines(self) -> list[str]:
        return [
            f"path: {self.path}",
            f"n_references: {len(self.references)}",
            f"total_reference_tks: {sum(len(ref) for ref in self.references)}",
            f"project: {self.project}",
            f"commit: {self.commit.summary() if self.commit else 'None'}",
        ]

    def stats(self) -> Mapping[str, int | float]:
        all_ref_tks = sum(len(x) for x in self.references)
        unchanged_ref_tks = sum(
            len(x) for name, x in self.named_references if "unchanged ref" in name
        )
        return {
            "input_tks": len(self.input_tks),
            "output_tks": len(self.output_tks),
            "n_references": len(self.references),
            "changed_reference_tks": all_ref_tks - unchanged_ref_tks,
            "unchanged_reference_tks": unchanged_ref_tks,
            "total_reference_tks": all_ref_tks,
        }


class C3TokenizerArgs(TypedDict):
    max_ref_tks: int
    max_query_tks: int
    max_output_tks: int
    max_scope_tks: int
    max_ref_tks_sum: int
    ref_chunk_overlap: int


@dataclass
class TsC3ProblemTokenizer:
    """
    ## Change log
    - 2.7: support `disable_builtin_defs`, `disable_unchanged_refs` and `current_code_only`.
    - 2.6: increase max_ref_tks_sum from 512 * 12 to 512 * 16.
    - 2.5: sort used references by path.
    - 2.4: encode each changed reference individually. Encode signatures for unchanged.
    """

    VERSION = "2.7"
    max_ref_tks: int = 512
    max_query_tks: int = 512
    max_output_tks: int = 256
    max_scope_tks: int = 128
    max_ref_tks_sum: int = 512 * 16
    ref_chunk_overlap: int = 32
    disable_builtin_defs: bool = True
    disable_unchanged_refs: bool = False
    # if true, use the current code instead of diff
    current_code_only: bool = False

    def get_args(self):
        return C3TokenizerArgs(
            max_ref_tks=self.max_ref_tks,
            max_query_tks=self.max_query_tks,
            max_output_tks=self.max_output_tks,
            max_scope_tks=self.max_scope_tks,
            max_ref_tks_sum=self.max_ref_tks_sum,
            ref_chunk_overlap=self.ref_chunk_overlap,
        )

    @classmethod
    def from_args(cls, args: C3TokenizerArgs) -> Self:
        return cls(**args)

    def __post_init__(self):
        self._offset_cache = dict[int, TkArray]()

    def tokenize_problem(
        self,
        problem: TsC3Problem,
    ) -> TkC3Problem:
        # 【guohx】如果设置为只使用当前代码，则将问题转换为当前代码版本
        if self.current_code_only:
            problem = _problem_to_current(problem)
        span = problem.span
        # 【guohx】获取原始的Token序列
        original: TokenSeq = span.original.tolist()
        tk_delta: TkDelta = span.delta
        # 【guohx】按行分割原始Token
        origin_lines = tk_splitlines(original)
        # 【guohx】获取需要编辑的行号并排序
        edit_lines = list(sorted(problem.edit_line_ids))
        # 【guohx】编辑的起始行
        edit_start = edit_lines[0]
        # 【guohx】编码作用域头部信息
        scope_tks = self._encode_headers(span.headers, 0)
        # 【guohx】计算主输入的最大Token数
        input_limit = self.max_query_tks - len(scope_tks)

        chunk_input = TokenSeq()  # 【guohx】主输入Token序列
        chunk_output = TokenSeq()  # 【guohx】期望输出Token序列
        last_line = edit_start  # 【guohx】上一次处理的行号

        # 【guohx】遍历需要编辑的行，最多处理N_Extra_Ids行
        for i, l in enumerate(edit_lines[:N_Extra_Ids]):
            # 【guohx】将last_line+1到l之间的所有行加入输入
            for line in origin_lines[last_line + 1 : l]:
                chunk_input.extend(line)
                chunk_input.append(Newline_id)

            # 【guohx】插入特殊的extra_id标记
            chunk_input.append(get_extra_id(i))
            # 【guohx】如果l在原始行范围内，将该行内容加入输入
            if l < len(origin_lines):
                chunk_input.extend(origin_lines[l])
                chunk_input.append(Newline_id)
                last_line = l
            # 【guohx】获取该行的变更内容，加入输出
            line_change = join_list(tk_delta.get_line_change(l), Newline_id)
            chunk_output.append(get_extra_id(i))
            chunk_output.extend(line_change)
            # 【guohx】如果变更内容不是删除结尾，补一个换行
            if line_change and line_change[-1] != Del_id:
                chunk_output.append(Newline_id)
            # 【guohx】如果输入超出限制，提前终止
            if len(chunk_input) > input_limit:
                break
        # 【guohx】计算编辑的终止行
        edit_stop = last_line + 1

        # 【guohx】限制输入Token数量
        chunk_input = truncate_section(
            chunk_input, TruncateAt.Right, input_limit, inplace=True
        )
        # 【guohx】限制输出Token数量
        chunk_output = truncate_output_tks(chunk_input, chunk_output)

        # 【guohx】处理编辑区上方的上下文
        above_tks = join_list(origin_lines[:edit_start] + [TokenSeq()], Newline_id)
        above_delta = tk_delta.for_input_range((0, edit_start))
        if self.current_code_only:
            above_tks = above_delta.apply_to_input(above_tks)
        else:
            above_tks = above_delta.apply_to_change(above_tks)
        # 【guohx】处理编辑区下方的上下文
        below_tks = join_list(origin_lines[edit_stop:] + [TokenSeq()], Newline_id)
        # 【guohx】尝试将部分上下文内联到输入中
        chunk_input, above_tks, below_tks = self._inline_some_context(
            chunk_input, above_tks, below_tks, input_limit
        )

        # 【guohx】再次限制输出Token数量
        chunk_output = truncate_section(
            chunk_output,
            TruncateAt.Right,
            self.max_output_tks,
            add_bos=False,
            inplace=True,
        )

        # 【guohx】将上方上下文分块
        above_chunks = break_into_chunks(
            above_tks,
            lambda i: self._encode_headers(span.headers, -1 - i),
            chunk_size=self.max_ref_tks,
            overlap=self.ref_chunk_overlap,
            right_to_left=True,
        )
        # 【guohx】将下方上下文分块
        if not below_tks:
            below_chunks = []
        else:
            below_chunks = break_into_chunks(
                below_tks,
                lambda i: self._encode_headers(span.headers, i + 1),
                chunk_size=self.max_ref_tks,
                overlap=self.ref_chunk_overlap,
            )
        # 【guohx】包装分块为命名引用
        above_chunks = [
            (f"above chunk {i}", TkArray.new(chunk))
            for i, chunk in enumerate(above_chunks)
        ]
        below_chunks = [
            (f"below chunk {i}", TkArray.new(chunk))
            for i, chunk in enumerate(below_chunks)
        ]
        # 【guohx】合并所有引用
        all_refs = above_chunks + below_chunks
        ref_size_sum = sum(len(ref) for _, ref in all_refs)

        truncated = False
        # 【guohx】如果引用总Token数未超限，继续添加未变更引用
        if ref_size_sum < self.max_ref_tks_sum:
            unchanged = problem.relevant_unchanged
            if self.disable_unchanged_refs:
                unchanged = {}
            if self.disable_builtin_defs:
                unchanged = {
                    k: v for k, v in unchanged.items() if not k.startswith("builtins.")
                }
            for i, chunk in enumerate(self._group_encode_unchanged_refs(unchanged)):
                all_refs.append((f"unchanged ref {i}", chunk))
                ref_size_sum += len(chunk)
        else:
            truncated = True

        # 【guohx】如果引用总Token数仍未超限，继续添加变更引用
        if ref_size_sum < self.max_ref_tks_sum:
            changed = self._group_encode_changed_refs(problem.relevant_changes)
            for i, chunk in enumerate(changed):
                all_refs.append((f"changed ref {i}", chunk))
                ref_size_sum += len(chunk)
        else:
            truncated = True

        # 【guohx】最终只保留未超限的引用
        ref_size_sum = 0
        kept_refs = list[tuple[str, TkArray]]()
        for name, ref in all_refs:
            if ref_size_sum + len(ref) > self.max_ref_tks_sum:
                truncated = True
                break
            ref_size_sum += len(ref)
            kept_refs.append((name, ref))

        # 【guohx】返回最终的Token化问题对象
        return TkC3Problem(
            TkArray.new(chunk_input),
            TkArray.new(scope_tks),
            TkArray.new(chunk_output),
            path=span.headers[-1].path,
            change_type=problem.change_type,
            named_references=kept_refs,
            project=problem.src_info["project"],
            commit=problem.src_info["commit"],
            truncated=truncated,
        )

    def _encode_headers(
        self, scope_changes: Sequence[ChangedHeader], offset: int
    ) -> TokenSeq:
        segs = [c.change_tks.tolist() for c in scope_changes]
        if offset != 0:
            segs.append(self._get_offset_tks(offset).tolist())
        segs.append([])
        scope_tks = join_list(segs, Newline_id)
        scope_tks = truncate_section(
            scope_tks, TruncateAt.Left, self.max_scope_tks, inplace=True
        )
        return scope_tks

    def _get_offset_tks(self, offset: int) -> TkArray:
        if (tks := self._offset_cache.get(offset)) is None:
            tks = TkArray.new(encode_single_line(f"# offset: {offset}"))
            self._offset_cache[offset] = tks
        return tks

    def _inline_some_context(
        self,
        input: TokenSeq,
        above_ctx: TokenSeq,
        below_ctx: TokenSeq,
        size_limit: int,
    ) -> tuple[TokenSeq, TokenSeq, TokenSeq]:
        "try move some some of the ctx tokens into the input if there's space."
        extra_space = size_limit - len(input)
        if (above_ctx or below_ctx) and extra_space > 0:
            truncated_above, truncated_below = truncate_sections(
                extra_space,
                (above_ctx, TruncateAt.Left),
                (below_ctx, TruncateAt.Right),
                add_bos=True,
            )
            above_left = len(above_ctx) - len(truncated_above)
            if above_left > 0:
                above_ctx = truncate_section(
                    above_ctx, TruncateAt.Right, above_left + self.ref_chunk_overlap
                )
            else:
                above_ctx = TokenSeq()
            below_left = len(below_ctx) - len(truncated_below)
            if below_left > 0:
                below_ctx = truncate_section(
                    below_ctx, TruncateAt.Left, below_left + self.ref_chunk_overlap
                )
            else:
                below_ctx = TokenSeq()
            input = truncated_above + input + truncated_below
        return input, above_ctx, below_ctx

    def _group_encode_unchanged_refs(
        self, elems: Mapping[PyFullName, TsDefinition]
    ) -> Sequence[TkArray]:
        def sort_key(e: TsDefinition):
            return (e.parent, min(e.start_locs, default=(0, 0)))

        results = list[TkArray]()
        this_chunk = TokenSeq()
        sorted_elems = [e for e in elems.values() if e.signatures and e.parent]
        sorted_elems.sort(key=sort_key)
        last_parent = None
        for defn in sorted_elems:
            parent = defn.parent
            header = f"at: {parent}\n" if parent != last_parent else ""
            text = header + indent("\n".join(s for s in defn.signatures), TAB) + "\n\n"
            tks = encode_lines_join(text)
            tks = truncate_section(
                tks, TruncateAt.Right, self.max_ref_tks, inplace=True
            )
            if len(this_chunk) + len(tks) > self.max_ref_tks:
                results.append(TkArray.new(this_chunk))
                this_chunk = tks
            else:
                this_chunk.extend(tks)
            last_parent = parent
        if this_chunk:
            results.append(TkArray.new(this_chunk))
        return results

    def _group_encode_changed_refs(
        self, changes: Sequence[ChangedCodeSpan]
    ) -> Sequence[TkArray]:
        return join_list(self._encode_changed_ref(x) for x in changes)

    def _encode_changed_ref(self, cspan: ChangedCodeSpan) -> Iterable[TkArray]:
        "encode a single changed reference"
        cspan_tks = cspan.delta.apply_to_change(cspan.original.tolist())
        for chunk in break_into_chunks(
            cspan_tks,
            lambda i: self._encode_headers(cspan.headers, i),
            self.max_ref_tks,
            overlap=self.ref_chunk_overlap,
        ):
            yield TkArray.new(chunk)

    def compute_stats(self, problems: Sequence[TsC3Problem]):
        all_stats = pmap(self._tokenize_stats, problems)
        if not all_stats:
            return dict()
        keys = all_stats[0].keys()
        stats = dict[str, dict]()
        for k in keys:
            stats[k] = scalar_stats([s[k] for s in all_stats])
        return stats

    def _tokenize_stats(self, problem: TsC3Problem):
        tkprob = self.tokenize_problem(problem)
        stats = dict(tkprob.stats())
        stats["input_cutoff"] = stats["input_tks"] >= self.max_query_tks
        stats["output_cutoff"] = stats["output_tks"] >= self.max_output_tks
        stats["reference_cutoff"] = tkprob.truncated
        return stats

    def __repr__(self):
        return repr_modified_args(self)

    @staticmethod
    def for_eval() -> "TsC3ProblemTokenizer":
        tkn = TsC3ProblemTokenizer()
        tkn.max_query_tks *= 2
        tkn.max_ref_tks *= 2
        tkn.max_ref_tks_sum *= 2
        return tkn


# Utils to convert code changes into the current version of the code


def _change_to_current(change_tks: TokenSeq) -> TokenSeq:
    new_lines = list[TokenSeq]()
    for line in tk_splitlines(change_tks):
        if line and line[0] == Add_id:
            new_lines.append(line[1:])
        elif line and line[0] == Del_id:
            pass
        else:
            new_lines.append(line)
    return join_list(new_lines, Newline_id)


def _header_to_current(header: ChangedHeader) -> ChangedHeader:
    tks = header.change_tks.tolist()
    change_tks = TkArray.new(_change_to_current(tks))
    return replace(header, change_tks=change_tks)


def _span_to_current(span: ChangedCodeSpan) -> ChangedCodeSpan:
    original = span.original.tolist()
    original = span.delta.apply_to_input(original)
    original = TkArray.new(original)
    delta = TkDelta.empty()
    headers = [_header_to_current(h) for h in span.headers]
    return replace(span, original=original, delta=delta, headers=headers)


def _problem_to_current(prob: TsC3Problem):
    span = prob.span
    original = span.original.tolist()
    delta = span.delta
    edit_line_ids = set(prob.edit_line_ids)
    assert edit_line_ids

    shift = 0
    new_lines = list[TokenSeq]()
    new_delta = dict[int, tuple]()
    new_edit_line_ids = set[int]()
    i = -1
    for i, line in enumerate(tk_splitlines(original)):
        if line and line[0] == Add_id:
            new_lines.append(line[1:])
        elif line and line[0] == Del_id:
            shift -= 1
        else:
            new_lines.append(line)
        if actions := delta.get_line_change(i):
            assert (i + shift) not in new_delta
            new_delta[i + shift] = actions
        if i in edit_line_ids:
            new_edit_line_ids.add(i + shift)
    if (i + 1) in edit_line_ids:
        new_edit_line_ids.add(i + shift + 1)

    if not new_edit_line_ids:
        print_err("original:")
        print_err(decode_tokens(original))
        raise ValueError(
            f"No edit lines left. {prob.edit_line_ids=}, {len(tk_splitlines(original))=}"
        )

    new_edit_line_ids = list(sorted(new_edit_line_ids))
    new_original = TkArray.new(join_list(new_lines, Newline_id))
    new_delta = TkDelta(new_delta)
    new_headers = [_header_to_current(h) for h in span.headers]
    new_span = replace(
        span, original=new_original, delta=new_delta, headers=new_headers
    )
    relevant_changes = [_span_to_current(c) for c in prob.relevant_changes]

    return replace(
        prob,
        span=new_span,
        edit_line_ids=new_edit_line_ids,
        relevant_changes=relevant_changes,
    )


def get_line_usages_ts(
    ts_module,  # TsParsedModule
    lines_to_analyze: Collection[int],
) -> LineUsageAnalysis:
    """
    遍历 tree-sitter AST，收集每一行的符号定义（函数、类、方法、变量）。
    """
    from tree_sitter import Node

    code = ts_module.code
    tree = ts_module.tree
    line2usages = {}

    def visit(node: Node, parent_name=""):
        # 只处理 function/class/method/variable
        if node.type in {"function_declaration", "class_declaration", "method_definition", "variable_declaration"}:
            # 计算全名
            name = ""
            for child in node.children:
                if child.type == "identifier":
                    name = code[child.start_byte:child.end_byte]
                    break
            full_name = parent_name + "." + name if parent_name else name
            # 记录定义
            defn = TsDefinition(full_name, {node.start_point}, set())
            defn.update(node, code)
            line = node.start_point[0] + 1  # tree-sitter 行号从0开始
            line2usages.setdefault(line, []).append(defn)
            # 递归处理子节点（如类里的方法）
            for child in node.children:
                visit(child, full_name)
        else:
            for child in node.children:
                visit(child, parent_name)

    visit(tree.root_node)
    # 只保留需要分析的行
    filtered = {l: v for l, v in line2usages.items() if l in lines_to_analyze}
    return LineUsageAnalysis(filtered)


def fix_ts_cache(cache_dir: Path):
    """
    为 TypeScript 版本设置缓存目录。
    这个函数类似于 Python 版本的 fix_jedi_cache，但用于 tree-sitter 缓存。
    """
    import os
    from pathlib import Path
    
    # 创建 tree-sitter 缓存目录
    ts_cache_dir = cache_dir / "tree_sitter_cache"
    ts_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置环境变量（如果 tree-sitter 支持的话）
    os.environ.setdefault("TREE_SITTER_CACHE_DIR", str(ts_cache_dir))
    
    # 创建其他可能需要的缓存目录
    parser_cache_dir = cache_dir / "parser_cache"
    parser_cache_dir.mkdir(parents=True, exist_ok=True)
    
    return ts_cache_dir
