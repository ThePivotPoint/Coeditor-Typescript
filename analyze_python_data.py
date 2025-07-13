#!/usr/bin/env python3
"""
【guohx】分析Python版本的processed数据文件内容
"""

import pickle
import sys
import os
from pathlib import Path

# 【guohx】添加项目路径
sys.path.insert(0, 'src')

def analyze_python_data():
    """【guohx】分析Python版本的processed数据"""
    
    processed_dir = Path("datasets_root/perm2k/processed/C3ProblemGenerator(VERSION=3.1, analyzer=())/")
    
    print("=" * 80)
    print("【guohx】Python版本 PROCESSED 数据分析")
    print("=" * 80)
    
    if not processed_dir.exists():
        print("❌ 目录不存在")
        return
    
    # 【guohx】遍历所有文件
    for file_path in processed_dir.iterdir():
        if file_path.is_file():
            print(f"\n📁 文件: {file_path.name}")
            print(f"📊 大小: {file_path.stat().st_size:,} bytes")
            print("-" * 60)
            
            try:
                # 【guohx】读取pickle文件
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                print(f"📋 数据类型: {type(data)}")
                
                if isinstance(data, dict):
                    print(f"🔑 字典键: {list(data.keys())}")
                    total_problems = 0
                    
                    for key, value in data.items():
                        print(f"\n  📂 分集: {key}")
                        print(f"    📊 类型: {type(value)}")
                        print(f"    📈 长度: {len(value) if hasattr(value, '__len__') else 'N/A'}")
                        
                        if hasattr(value, '__len__') and len(value) > 0:
                            total_problems += len(value)
                            
                            # 【guohx】分析第一个问题
                            first_problem = value[0]
                            print(f"    🔍 第一个问题类型: {type(first_problem)}")
                            
                            # 【guohx】如果是C3Problem，显示其属性
                            if hasattr(first_problem, '__dict__'):
                                print(f"    📝 属性: {list(first_problem.__dict__.keys())}")
                                
                                # 【guohx】显示一些关键属性
                                for attr in ['repo_name', 'commit_hash', 'file_path', 'edit_type']:
                                    if hasattr(first_problem, attr):
                                        attr_value = getattr(first_problem, attr)
                                        print(f"      {attr}: {attr_value}")
                
                elif isinstance(data, list):
                    print(f"📈 列表长度: {len(data)}")
                    if len(data) > 0:
                        first_item = data[0]
                        print(f"🔍 第一个元素类型: {type(first_item)}")
                        if hasattr(first_item, '__dict__'):
                            print(f"📝 属性: {list(first_item.__dict__.keys())}")
                
                print(f"\n✅ 总问题数量: {total_problems if 'total_problems' in locals() else 'N/A'}")
                
            except Exception as e:
                print(f"❌ 读取错误: {e}")
                import traceback
                traceback.print_exc()
            
            print("=" * 80)

def analyze_single_problem():
    """【guohx】详细分析单个问题"""
    
    print("\n" + "=" * 80)
    print("【guohx】详细分析单个问题")
    print("=" * 80)
    
    # 【guohx】选择一个较小的文件进行分析
    file_path = Path("datasets_root/perm2k/processed/C3ProblemGenerator(VERSION=3.1, analyzer=())/deepseek-ai~DeepSeek-V3(1000, is_training=False)")
    
    if not file_path.exists():
        print("❌ 文件不存在")
        return
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict) and len(data) > 0:
            # 【guohx】获取第一个分集的第一个问题
            first_split = list(data.keys())[0]
            first_problem = data[first_split][0]
            
            print(f"📂 分集: {first_split}")
            print(f"🔍 问题类型: {type(first_problem)}")
            print(f"📝 所有属性:")
            
            for attr_name, attr_value in first_problem.__dict__.items():
                print(f"  {attr_name}: {type(attr_value)} = {attr_value}")
                
    except Exception as e:
        print(f"❌ 分析错误: {e}")

if __name__ == "__main__":
    analyze_python_data()
    analyze_single_problem() 