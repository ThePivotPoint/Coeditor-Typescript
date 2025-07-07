"""
TypeScript 数据集处理模块
处理 TypeScript 项目的代码分析和问题生成
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from dataclasses import asdict

from coeditor.common import *
from coeditor_ts.c3problem_ts import TSC3ProblemGenerator


class TypeScriptDatasetProcessor:
    """TypeScript 数据集处理器"""
    
    def __init__(self, dataset_dir: Path, max_workers: int = 4):
        self.dataset_dir = dataset_dir
        self.max_workers = max_workers
        self.generator = TSC3ProblemGenerator()
        
    def process_full_dataset(self):
        """处理完整数据集"""
        print("开始处理完整 TypeScript 数据集...")
        
        # 处理各个分割
        for split in ["train", "valid", "test"]:
            print(f"\n处理 {split} 分割...")
            self._process_split(split)
    
    def process_test_mode(self):
        """测试模式，只处理少量数据"""
        print("开始测试模式处理...")
        
        # 只处理 test 分割的少量数据
        split = "test"
        print(f"处理 {split} 分割（测试模式）...")
        self._process_split(split, test_mode=True)
    
    def _process_split(self, split: str, test_mode: bool = False):
        """处理单个分割"""
        repos_dir = self.dataset_dir / "repos" / split
        
        if not repos_dir.exists():
            print(f"警告: 分割目录不存在 {repos_dir}")
            return
        
        # 获取项目列表
        projects = [p for p in repos_dir.iterdir() if p.is_dir()]
        
        if test_mode:
            # 测试模式只处理前2个项目
            projects = projects[:2]
            print(f"测试模式: 只处理 {len(projects)} 个项目")
        
        print(f"找到 {len(projects)} 个项目")
        
        # 并行处理项目
        results = pmap(
            self._process_single_project,
            projects,
            desc=f"处理 {split} 项目",
            max_workers=self.max_workers,
            chunksize=1
        )
        
        # 收集结果
        all_problems = []
        project_stats = {}
        
        for project_path, problems in results:
            if problems:
                project_name = project_path.name
                project_stats[project_name] = {
                    'problem_count': len(problems),
                    'project_path': str(project_path)
                }
                all_problems.extend(problems)
        
        # 保存结果
        output_dir = self.dataset_dir / "processed" / split
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存问题数据
         # 保存问题数据
        problems_file = output_dir / "problems.json"
        problems_data = [asdict(prob) for prob in all_problems]
        with open(problems_file, 'w', encoding='utf-8') as f:
            json.dump(problems_data, f, indent=2, ensure_ascii=False)
        
        # 保存统计信息
        stats_file = output_dir / "stats.json"
        stats = {
            'total_projects': len(projects),
            'processed_projects': len(project_stats),
            'total_problems': len(all_problems),
            'project_stats': project_stats
        }
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"{split} 分割处理完成:")
        print(f"  总项目数: {len(projects)}")
        print(f"  成功处理项目数: {len(project_stats)}")
        print(f"  总问题数: {len(all_problems)}")
        print(f"  结果保存到: {output_dir}")
    
    def _process_single_project(self, project_path: Path) -> Tuple[Path, List]:
        """处理单个项目"""
        try:
            # 检查是否为有效的 TypeScript 项目
            if not self._is_valid_ts_project(project_path):
                print(f"跳过无效项目: {project_path.name}")
                return project_path, []
            
            # 生成问题
            problems = self.generator.generate_from_project(project_path)
            
            print(f"项目 {project_path.name}: 生成 {len(problems)} 个问题")
            return project_path, problems
            
        except Exception as e:
            print(f"处理项目 {project_path.name} 时出错: {e}")
            return project_path, []
    
    def _is_valid_ts_project(self, project_path: Path) -> bool:
        """检查是否为有效的 TypeScript 项目"""
        # 检查是否有 package.json
        package_json = project_path / "package.json"
        if not package_json.exists():
            return False
        
        # 检查是否有 TypeScript 文件
        ts_files = self._find_ts_files(project_path)
        if not ts_files:
            return False
        
        return True
    
    def _find_ts_files(self, directory: Path) -> List[Path]:
        """查找 TypeScript 文件"""
        ts_files = []
        for root, dirs, files in os.walk(directory):
            # 跳过 node_modules 和 .git 目录
            dirs[:] = [d for d in dirs if d not in ['node_modules', '.git']]
            
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in ['.ts', '.tsx']:
                    ts_files.append(file_path)
        
        return ts_files


def process_ts_projects(projects_root: Path, generator: TSC3ProblemGenerator, 
                       split: str = "train") -> Dict[str, List]:
    """处理 TypeScript 项目（兼容旧接口）"""
    processor = TypeScriptDatasetProcessor(projects_root.parent.parent)
    processor._process_split(split)
    return {}  # 简化返回


def get_repo_signature(repo_path: Path) -> str:
    """获取仓库签名（简化版本）"""
    # 对于 TypeScript 项目，我们使用 package.json 的哈希作为签名
    package_json = repo_path / "package.json"
    if package_json.exists():
        # 简化的哈希计算
        with open(package_json, 'rb') as f:
            return str(hash(f.read()))
    else:
        # 如果没有 package.json，使用目录名的哈希
        return str(hash(repo_path.name)) 