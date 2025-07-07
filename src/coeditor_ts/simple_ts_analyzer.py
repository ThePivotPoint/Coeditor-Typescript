#!/usr/bin/env python3
"""
简化的 TypeScript 静态分析器，用于替换 JediUsageAnalyzer
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Set, Collection
from dataclasses import dataclass

@dataclass
class TSDefinition:
    """TypeScript 定义信息，对应 PyDefinition"""
    full_name: str
    start_locs: Set[tuple[int, int]]
    signatures: Set[str]
    
    def __post_init__(self):
        self.parent = ".".join(self.full_name.split(".")[:-1])

@dataclass
class TSLineUsageAnalysis:
    """TypeScript 行使用分析，对应 LineUsageAnalysis"""
    line2usages: Dict[int, List[TSDefinition]]
    
    def __repr__(self):
        lines = ["TSLineUsageAnalysis("]
        for line, usages in self.line2usages.items():
            lines.append(f"    {line}: {usages}")
        lines.append(")")
        return "\n".join(lines)

class TypeScriptUsageAnalyzer:
    """TypeScript 使用分析器，对应 JediUsageAnalyzer"""
    
    def __init__(self, include_parent_usages: bool = True, include_builtins: bool = False):
        self.include_parent_usages = include_parent_usages
        self.include_builtins = include_builtins
        self.error_counts = {}
        
        # TypeScript 关键字
        self.keywords = {
            'const', 'let', 'var', 'function', 'class', 'interface', 'type', 
            'import', 'export', 'if', 'else', 'for', 'while', 'return', 
            'new', 'this', 'super', 'async', 'await', 'true', 'false', 
            'null', 'undefined', 'NaN', 'Infinity', 'string', 'number', 
            'boolean', 'void', 'any', 'unknown', 'never', 'object', 'array'
        }
    
    def get_line_usages(self, ts_file_path: str, lines_to_analyze: Collection[int], silent: bool = False) -> TSLineUsageAnalysis:
        """分析 TypeScript 文件中指定行的使用情况"""
        try:
            with open(ts_file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            lines = source_code.split('\n')
            line2usages = {}
            
            for line_num in lines_to_analyze:
                if 1 <= line_num <= len(lines):
                    line = lines[line_num - 1]
                    identifiers = self._extract_identifiers_from_line(line)
                    
                    if identifiers:
                        definitions = []
                        for identifier in identifiers:
                            # 创建简化的定义信息
                            definition = TSDefinition(
                                full_name=identifier,
                                start_locs={(line_num, 0)},  # 简化的位置信息
                                signatures={f"{identifier}"}
                            )
                            definitions.append(definition)
                        
                        line2usages[line_num] = definitions
            
            return TSLineUsageAnalysis(line2usages)
            
        except Exception as e:
            if not silent:
                print(f"Error analyzing {ts_file_path}: {e}")
            self.add_error(str(e))
            return TSLineUsageAnalysis({})
    
    def _extract_identifiers_from_line(self, line: str) -> List[str]:
        """从一行代码中提取标识符"""
        # 简单的标识符正则表达式
        identifier_pattern = r'\b[a-zA-Z_$][a-zA-Z0-9_$]*\b'
        matches = re.findall(identifier_pattern, line)
        
        # 过滤掉关键字
        identifiers = [match for match in matches if match not in self.keywords]
        
        return identifiers
    
    def add_error(self, error_text: str):
        """记录错误"""
        self.error_counts[error_text] = self.error_counts.get(error_text, 0) + 1

# 测试函数
def test_analyzer():
    """测试 TypeScript 分析器"""
    analyzer = TypeScriptUsageAnalyzer()
    
    # 创建测试文件
    test_code = """
interface User {
    id: number;
    name: string;
}

class UserService {
    private users: User[] = [];
    
    addUser(user: User): void {
        this.users.push(user);
    }
    
    getUserById(id: number): User | undefined {
        return this.users.find(user => user.id === id);
    }
}

const userService = new UserService();
const user: User = {
    id: 1,
    name: "John"
};

userService.addUser(user);
"""
    
    test_file = "test_sample.ts"
    with open(test_file, 'w') as f:
        f.write(test_code)
    
    # 分析文件
    lines_to_analyze = [3, 6, 9, 12, 15, 18, 21, 24, 27]
    result = analyzer.get_line_usages(test_file, lines_to_analyze)
    
    print("分析结果:")
    print(result)
    
    # 清理测试文件
    Path(test_file).unlink(missing_ok=True)

if __name__ == "__main__":
    test_analyzer() 