"""
TypeScript 版本的 C3Problem 生成器
"""

import os
import pickle
import warnings
import traceback
from pathlib import Path
from typing import Dict, List, Any, Sequence, Collection, Mapping
from dataclasses import dataclass, field
from tqdm import tqdm

# 导入我们的 TypeScript 分析器
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simple_ts_analyzer import TypeScriptUsageAnalyzer, TSLineUsageAnalysis, TSDefinition

@dataclass
class TSChangedHeader:
    """TypeScript 变更头部，对应 ChangedHeader"""
    change_tks: List[str]  # 简化的 token 表示
    type: str
    line_range: tuple[int, int]
    path: str

@dataclass
class TSChangedCodeSpan:
    """TypeScript 变更代码段，对应 ChangedCodeSpan"""
    headers: Sequence[TSChangedHeader]
    original: List[str]
    delta: Dict[str, Any]  # 简化的变更表示
    line_range: tuple[int, int]
    module: str

@dataclass
class TSSrcInfo:
    """TypeScript 源信息，对应 SrcInfo"""
    project: str
    commit: str | None

@dataclass
class TSC3Problem:
    """TypeScript 版本的 C3Problem"""
    span: TSChangedCodeSpan
    edit_line_ids: Sequence[int]
    relevant_changes: Sequence[TSChangedCodeSpan]
    relevant_unchanged: Mapping[str, TSDefinition]
    change_type: str
    src_info: TSSrcInfo
    transformations: tuple[str, ...] = ()

@dataclass
class TSC3PreAnalysis:
    """TypeScript 预分析结果"""
    training_samples: set[tuple[str, tuple[int, int]]]
    usage_analysis: Mapping[str, TSLineUsageAnalysis]

@dataclass
class TSC3ProblemGenerator:
    """TypeScript 版本的 C3ProblemGenerator"""
    
    VERSION = "1.0"
    max_span_lines: int = 500
    max_span_chars: int = 6000
    neg_to_pos_ratio: float = 0.0
    analyzer: TypeScriptUsageAnalyzer = field(default_factory=TypeScriptUsageAnalyzer)
    
    def __post_init__(self):
        self._is_training = False
        self.error_counts = {}
    
    def set_training(self, is_training: bool) -> None:
        self._is_training = is_training
    
    @property
    def is_training(self) -> bool:
        return getattr(self, "_is_training", False)
    
    def use_unchanged(self) -> bool:
        return self.neg_to_pos_ratio > 0
    
    def pre_edit_analysis(
        self,
        pstate: Dict[str, Any],  # 简化的项目状态
        modules: Dict[str, Any],  # 简化的模块映射
        changes: Dict[str, Any],  # 简化的变更映射
    ) -> TSC3PreAnalysis:
        """预编辑分析，对应原来的 pre_edit_analysis"""
        
        # 简化的实现：选择一些训练样本
        selected_set = set()
        negative_set = set()
        
        for mname, mchange in changes.items():
            # 这里简化处理，实际应该遍历变更的代码段
            if isinstance(mchange, dict) and 'changed' in mchange:
                for cspan in mchange['changed']:
                    if self.should_mk_problem(cspan, func_only=not self.is_training):
                        if cspan.get('type') == 'modified':
                            if cspan.get('unchanged'):
                                negative_set.add((mname, cspan.get('line_range', (0, 0))))
                            else:
                                selected_set.add((mname, cspan.get('line_range', (0, 0))))
        
        # 包含一些负样本
        if negative_set:
            select_prob = len(selected_set) * self.neg_to_pos_ratio / len(negative_set)
            for x in negative_set:
                if hash(str(x)) % 100 < select_prob * 100:  # 简化的随机选择
                    selected_set.add(x)
        
        # 简化的使用分析
        usages = {}
        for mname, mchange in changes.items():
            usages[mname] = TSLineUsageAnalysis({})
            
            # 这里应该分析实际的代码行，现在简化处理
            if isinstance(mchange, dict) and 'file_path' in mchange:
                try:
                    lines_to_analyze = {1, 2, 3}  # 简化的行号
                    line_usages = self.analyzer.get_line_usages(
                        mchange['file_path'], lines_to_analyze, silent=True
                    )
                    usages[mname] = line_usages
                except Exception as e:
                    self.add_error(str(e))
        
        return TSC3PreAnalysis(
            training_samples=selected_set,
            usage_analysis=usages,
        )
    
    def post_edit_analysis(
        self,
        pstate: Dict[str, Any],
        modules: Dict[str, Any],
        changes: Dict[str, Any],
    ) -> List[str]:
        """后编辑分析，返回模块的拓扑顺序"""
        # 简化的实现：返回所有模块名
        return list(changes.keys())
    
    def process_change(
        self,
        pchange: Dict[str, Any],  # 简化的项目变更
        pre_analysis: TSC3PreAnalysis,
        module_order: Sequence[str],
    ) -> Sequence[TSC3Problem]:
        """处理变更，生成 C3Problem 列表"""
        
        problems = []
        
        for m in module_order:
            if m not in pchange.get('changed', {}):
                continue
            
            mchange = pchange['changed'][m]
            usages = pre_analysis.usage_analysis.get(m, TSLineUsageAnalysis({}))
            
            # 简化的实现：为每个变更创建一个问题
            if isinstance(mchange, dict) and 'changed' in mchange:
                for span in mchange['changed']:
                    if (m, span.get('line_range', (0, 0))) in pre_analysis.training_samples:
                        # 创建简化的代码段
                        code_span = TSChangedCodeSpan(
                            headers=[TSChangedHeader([], "function", (0, 0), m)],
                            original=span.get('original', []),
                            delta=span.get('delta', {}),
                            line_range=span.get('line_range', (0, 0)),
                            module=m
                        )
                        
                        # 创建问题
                        line_range = span.get('line_range', (0, 1))
                        start_line, end_line = line_range
                        edit_line_count = max(1, end_line - start_line)
                        problem = TSC3Problem(
                            span=code_span,
                            edit_line_ids=list(range(edit_line_count)),
                            relevant_changes=[],
                            relevant_unchanged={},
                            change_type=span.get('type', 'modified'),
                            src_info=TSSrcInfo(
                                project=pchange.get('project_name', 'unknown'),
                                commit=pchange.get('commit_info', None)
                            )
                        )
                        
                        problems.append(problem)
        
        return problems
    
    def should_mk_problem(self, span: Dict[str, Any], func_only: bool, max_chars: int = None, max_lines: int = None) -> bool:
        """判断是否应该为这个变更创建问题"""
        if max_chars is None:
            max_chars = self.max_span_chars
        if max_lines is None:
            max_lines = self.max_span_lines
        
        # 简化的判断逻辑
        span_type = span.get('type', '')
        if span_type != 'modified':
            return False
        
        original = span.get('original', [])
        if len(str(original)) > max_chars:
            return False
        
        line_range = span.get('line_range', (0, 0))
        if line_range[1] - line_range[0] > max_lines:
            return False
        
        return True
    
    def add_error(self, error_text: str):
        """记录错误"""
        self.error_counts[error_text] = self.error_counts.get(error_text, 0) + 1
    
    def clear_stats(self) -> None:
        """清除统计信息"""
        self.error_counts.clear()
    
    def append_stats(self, stats: Dict[str, Any]) -> None:
        """添加统计信息"""
        if 'analyzer_errors' not in stats:
            stats['analyzer_errors'] = {}
        stats['analyzer_errors'].update(self.error_counts)
    
    def generate_from_project(self, project_path: Path) -> List[TSC3Problem]:
        """从项目生成 C3Problem 列表"""
        problems = []
        
        try:
            # 查找 TypeScript 文件
            ts_files = []
            for root, dirs, files in os.walk(project_path):
                # 跳过 node_modules 和 .git 目录
                dirs[:] = [d for d in dirs if d not in ['node_modules', '.git']]
                
                for file in files:
                    file_path = Path(root) / file
                    if file_path.suffix.lower() in ['.ts', '.tsx']:
                        ts_files.append(file_path)
            
            if not ts_files:
                print(f"项目 {project_path.name} 中没有找到 TypeScript 文件")
                return problems
            
            # 处理每个 TypeScript 文件
            for file_path in ts_files[:5]:  # 限制处理文件数量
                try:
                    file_problems = self._generate_from_file(file_path, project_path)
                    problems.extend(file_problems)
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")
                    continue
            
        except Exception as e:
            print(f"处理项目 {project_path.name} 时出错: {e}")
        
        return problems
    
    def _generate_from_file(self, file_path: Path, project_path: Path) -> List[TSC3Problem]:
        """从单个文件生成 C3Problem 列表"""
        problems = []
        
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if not lines:
                return problems
            
            # 简化的实现：为文件的每个函数创建一个问题
            # 这里我们假设每个函数大约10行代码
            for i in range(0, len(lines), 10):
                if i + 10 > len(lines):
                    break
                
                # 创建简化的变更表示
                original_lines = lines[i:i+10]
                line_range = (i + 1, i + 10)
                
                # 创建代码段
                code_span = TSChangedCodeSpan(
                    headers=[TSChangedHeader([], "function", line_range, str(file_path))],
                    original=original_lines,
                    delta={'type': 'modify'},
                    line_range=line_range,
                    module=str(file_path.relative_to(project_path))
                )
                
                # 创建问题
                problem = TSC3Problem(
                    span=code_span,
                    edit_line_ids=list(range(10)),
                    relevant_changes=[],
                    relevant_unchanged={},
                    change_type='modified',
                    src_info=TSSrcInfo(
                        project=project_path.name,
                        commit=None
                    )
                )
                
                problems.append(problem)
        
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
        
        return problems

# 测试函数
def test_c3_generator():
    """测试 C3ProblemGenerator"""
    generator = TSC3ProblemGenerator()
    generator.set_training(True)
    
    # 模拟项目状态和变更
    pstate = {}
    modules = {'test_module': {'file_path': 'test_sample.ts'}}
    changes = {
        'test_module': {
            'changed': [
                {
                    'type': 'modified',
                    'original': ['const x = 1;'],
                    'delta': {'type': 'modify'},
                    'line_range': (1, 2),
                    'unchanged': False
                }
            ],
            'file_path': 'test_sample.ts'
        }
    }
    
    # 执行预分析
    pre_analysis = generator.pre_edit_analysis(pstate, modules, changes)
    print("预分析结果:", pre_analysis)
    
    # 执行后分析
    module_order = generator.post_edit_analysis(pstate, modules, changes)
    print("模块顺序:", module_order)
    
    # 模拟项目变更
    pchange = {
        'project_name': 'test_project',
        'commit_info': 'test_commit',
        'changed': changes
    }
    
    # 处理变更
    problems = generator.process_change(pchange, pre_analysis, module_order)
    print(f"生成了 {len(problems)} 个问题")
    
    for i, problem in enumerate(problems):
        print(f"问题 {i+1}: {problem}")

if __name__ == "__main__":
    test_c3_generator() 

# === Add stubs for missing symbols if not present ===
from dataclasses import dataclass

@dataclass
class TSC3ProblemChangeInlining:
    max_inline_ratio: float = 0.6
    allow_empty_problems: bool = True
    def transform(self, problems):
        return problems

@dataclass
class TSC3ProblemTokenizer:
    def compute_stats(self, problems):
        return {"n_problems": len(problems)} 