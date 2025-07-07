#!/usr/bin/env python3
"""
测试 TypeScript 数据处理流程
"""

import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from coeditor_ts.ts_dataset import TypeScriptDatasetProcessor
from coeditor_ts.c3problem_ts import TSC3ProblemGenerator


def test_ts_generator():
    """测试 TypeScript C3Problem 生成器"""
    print("测试 TypeScript C3Problem 生成器...")
    
    generator = TSC3ProblemGenerator()
    
    # 创建测试 TypeScript 项目
    test_project = Path("test_ts_project")
    test_project.mkdir(exist_ok=True)
    
    # 创建测试 TypeScript 文件
    test_file = test_project / "main.ts"
    test_content = """
function add(a: number, b: number): number {
    return a + b;
}

function multiply(a: number, b: number): number {
    return a * b;
}

interface Calculator {
    add(a: number, b: number): number;
    multiply(a: number, b: number): number;
}

class SimpleCalculator implements Calculator {
    add(a: number, b: number): number {
        return add(a, b);
    }
    
    multiply(a: number, b: number): number {
        return multiply(a, b);
    }
}

export { add, multiply, Calculator, SimpleCalculator };
"""
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    # 创建 package.json
    package_json = test_project / "package.json"
    package_content = {
        "name": "test-ts-project",
        "version": "1.0.0",
        "main": "main.ts",
        "scripts": {
            "build": "tsc"
        }
    }
    
    import json
    with open(package_json, 'w', encoding='utf-8') as f:
        json.dump(package_content, f, indent=2)
    
    # 测试生成问题
    problems = generator.generate_from_project(test_project)
    
    print(f"生成了 {len(problems)} 个问题")
    for i, problem in enumerate(problems[:3]):  # 只显示前3个
        print(f"问题 {i+1}:")
        print(f"  项目: {problem.src_info.project}")
        print(f"  变更类型: {problem.change_type}")
        print(f"  行范围: {problem.span.line_range}")
        print(f"  原始代码行数: {len(problem.span.original)}")
        print()
    
    # 清理测试文件
    import shutil
    shutil.rmtree(test_project)
    
    return len(problems) > 0


def test_dataset_processor():
    """测试数据集处理器"""
    print("\n测试数据集处理器...")
    
    # 创建测试数据集目录
    test_dataset = Path("test_dataset")
    test_dataset.mkdir(exist_ok=True)
    
    # 创建测试项目结构
    repos_dir = test_dataset / "repos" / "test"
    repos_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制测试项目
    test_project = Path("test_ts_project")
    if test_project.exists():
        import shutil
        shutil.copytree(test_project, repos_dir / "test-project", dirs_exist_ok=True)
    
    # 创建处理器
    processor = TypeScriptDatasetProcessor(test_dataset, max_workers=1)
    
    # 测试处理
    try:
        processor.process_test_mode()
        print("数据集处理测试完成")
        return True
    except Exception as e:
        print(f"数据集处理测试失败: {e}")
        return False
    finally:
        # 清理测试文件
        import shutil
        if test_dataset.exists():
            shutil.rmtree(test_dataset)


def main():
    """主测试函数"""
    print("开始测试 TypeScript 数据处理流程...")
    
    # 测试生成器
    generator_ok = test_ts_generator()
    
    # 测试数据集处理器
    processor_ok = test_dataset_processor()
    
    print("\n测试结果:")
    print(f"生成器测试: {'通过' if generator_ok else '失败'}")
    print(f"处理器测试: {'通过' if processor_ok else '失败'}")
    
    if generator_ok and processor_ok:
        print("\n所有测试通过！TypeScript 数据处理流程正常工作。")
    else:
        print("\n部分测试失败，请检查错误信息。")


if __name__ == "__main__":
    main() 