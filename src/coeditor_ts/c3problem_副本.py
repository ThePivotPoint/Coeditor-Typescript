"""
TypeScript C3 problem generation and analysis module.
"""

from dataclasses import dataclass, field
from typing import *
from pathlib import Path

import tree_sitter
import tree_sitter_typescript

from .common import *
from .scoped_changes import *
from coeditor.encoding import (
    tk_splitlines,
    N_Extra_Ids,
    Newline_id,
    get_extra_id,
    Del_id,
    truncate_section,
    TruncateAt,
    join_list,
    truncate_output_tks,
    break_into_chunks,
)
# 可能还需要：
# from coeditor.encoding import join_list, truncate_output_tks

# TypeScript Definition
@dataclass
class TsDefinition:
    full_name: TsFullName
    start_locs: set[tuple[int, int]] = field(default_factory=set)
    signatures: set[str] = field(default_factory=set)
    type_info: str = ""

# 用法分析器（骨架）
@dataclass
class TsUsageAnalyzer:
    def get_line_usages(self, tree: TsTree, lines_to_analyze: Collection[int], silent: bool = False):
        # TODO: 实现 TypeScript 用法分析
        return {}

# TypeScript C3Problem
@dataclass(frozen=True)
class TsC3Problem:
    "Contextual code change prediction problem for TypeScript."
    span: Any
    edit_line_ids: Sequence[int]
    relevant_changes: Sequence[Any]
    relevant_unchanged: Mapping[TsFullName, TsDefinition]
    change_type: Any
    src_info: dict
    transformations: tuple = ()

    def summary(self) -> str:
        return f"TypeScript C3Problem: {self.span}"

# TypeScript C3ProblemGenerator（骨架）

# TypeScript C3ProblemTokenizer（骨架）
@dataclass
class TsC3ProblemTokenizer:
    VERSION = "0.1"
    max_query_tks: int = 512
    max_output_tks: int = 256
    max_ref_tks: int = 512
    max_ref_tks_sum: int = 512 * 16
    ref_chunk_overlap: int = 32
    disable_unchanged_refs: bool = False
    disable_builtin_defs: bool = False
    current_code_only: bool = False

    def tokenize_problem(self, problem: TsC3Problem):
        """
        Tokenize a TypeScript C3 problem into input, output, and references.
        参考Python tokenize_problem方法实现，逐步处理主输入、输出、上下文引用。
        """
        # TODO: 这里假设TsC3Problem结构与C3Problem类似，后续可根据TS具体结构调整
        if self.current_code_only:
            # TODO: 实现TypeScript的 _problem_to_current
            pass
        span = problem.span
        # 假设span.original/tk_delta等价于Python实现
        original = span.original.tolist()
        tk_delta = span.delta
        origin_lines = tk_splitlines(original)
        edit_lines = list(sorted(problem.edit_line_ids))
        edit_start = edit_lines[0]
        # 编码作用域头部信息
        scope_tks = self._encode_headers(span.headers, 0)
        input_limit = self.max_query_tks - len(scope_tks)

        chunk_input = []  # 主输入Token序列
        chunk_output = []  # 期望输出Token序列
        last_line = edit_start  # 上一次处理的行号

        # 遍历需要编辑的行，最多处理N_Extra_Ids行
        for i, l in enumerate(edit_lines[:N_Extra_Ids]):
            # 将last_line+1到l之间的所有行加入输入
            for line in origin_lines[last_line + 1 : l]:
                chunk_input.extend(line)
                chunk_input.append(Newline_id)
            # 插入特殊的extra_id标记
            chunk_input.append(get_extra_id(i))
            # 如果l在原始行范围内，将该行内容加入输入
            if l < len(origin_lines):
                chunk_input.extend(origin_lines[l])
                chunk_input.append(Newline_id)
                last_line = l
            # 获取该行的变更内容，加入输出
            line_change = join_list(tk_delta.get_line_change(l), Newline_id)
            chunk_output.append(get_extra_id(i))
            chunk_output.extend(line_change)
            # 如果变更内容不是删除结尾，补一个换行
            if line_change and line_change[-1] != Del_id:
                chunk_output.append(Newline_id)
            # 如果输入超出限制，提前终止
            if len(chunk_input) > input_limit:
                break
        # 计算编辑的终止行
        edit_stop = last_line + 1

        # 限制输入Token数量
        chunk_input = truncate_section(chunk_input, TruncateAt.Right, input_limit, inplace=True)
        # 限制输出Token数量
        chunk_output = truncate_output_tks(chunk_input, chunk_output)

        # 处理编辑区上方的上下文
        above_tks = join_list(origin_lines[:edit_start] + [[]], Newline_id)
        above_delta = tk_delta.for_input_range((0, edit_start))
        if self.current_code_only:
            above_tks = above_delta.apply_to_input(above_tks)
        else:
            above_tks = above_delta.apply_to_change(above_tks)
        # 处理编辑区下方的上下文
        below_tks = join_list(origin_lines[edit_stop:] + [[]], Newline_id)
        # 尝试将部分上下文内联到输入中
        chunk_input, above_tks, below_tks = self._inline_some_context(chunk_input, above_tks, below_tks, input_limit)

        # 再次限制输出Token数量
        chunk_output = truncate_section(chunk_output, TruncateAt.Right, self.max_output_tks, add_bos=False, inplace=True)

        # 将上方上下文分块
        above_chunks = break_into_chunks(
            above_tks,
            lambda i: self._encode_headers(span.headers, -1 - i),
            chunk_size=self.max_ref_tks,
            overlap=self.ref_chunk_overlap,
            right_to_left=True,
        )
        # 将下方上下文分块
        if not below_tks:
            below_chunks = []
        else:
            below_chunks = break_into_chunks(
                below_tks,
                lambda i: self._encode_headers(span.headers, i + 1),
                chunk_size=self.max_ref_tks,
                overlap=self.ref_chunk_overlap,
            )
        # 包装分块为命名引用
        above_chunks = [
            (f"above chunk {i}", chunk)
            for i, chunk in enumerate(above_chunks)
        ]
        below_chunks = [
            (f"below chunk {i}", chunk)
            for i, chunk in enumerate(below_chunks)
        ]
        # 合并所有引用
        all_refs = above_chunks + below_chunks
        ref_size_sum = sum(len(ref) for _, ref in all_refs)

        truncated = False
        # 如果引用总Token数未超限，继续添加未变更引用
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

        # 如果引用总Token数仍未超限，继续添加变更引用
        if ref_size_sum < self.max_ref_tks_sum:
            changed = self._group_encode_changed_refs(problem.relevant_changes)
            for i, chunk in enumerate(changed):
                all_refs.append((f"changed ref {i}", chunk))
                ref_size_sum += len(chunk)
        else:
            truncated = True

        # 最终只保留未超限的引用
        ref_size_sum = 0
        kept_refs = []
        for name, ref in all_refs:
            if ref_size_sum + len(ref) > self.max_ref_tks_sum:
                truncated = True
                break
            ref_size_sum += len(ref)
            kept_refs.append((name, ref))

        # 返回最终的Token化问题对象
        return {
            "input": chunk_input,
            "scope": scope_tks,
            "output": chunk_output,
            "references": kept_refs,
            "truncated": truncated,
        }

    # 下面的辅助方法需要根据TS实现
    def _encode_headers(self, scope_changes, offset):
        # TODO: 实现TS作用域头部编码
        return []
    def _inline_some_context(self, input, above_ctx, below_ctx, size_limit):
        # TODO: 实现上下文内联逻辑
        return input, above_ctx, below_ctx
    def _group_encode_unchanged_refs(self, elems):
        # TODO: 实现未变更引用分组编码
        return []
    def _group_encode_changed_refs(self, changes):
        # TODO: 实现变更引用分组编码
        return []

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
        stats = dict()
        # stats["input_tks"] = len(tkprob["input"])
        # stats["output_tks"] = len(tkprob["output"])
        # stats["n_references"] = len(tkprob["references"])
        # stats["reference_cutoff"] = tkprob["truncated"]
        return stats
class TsC3ProblemTransform(ABC):
    "A strategy to generate new C3 problems from the orginal ones."

    @abstractmethod
    def transform(self, prob: TsC3Problem) -> Sequence[TsC3Problem]:
        ...
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

    def transform(self, prob: C3Problem) -> Sequence[C3Problem]:
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

        prob_count = list[tuple[C3Problem, int]]()
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

