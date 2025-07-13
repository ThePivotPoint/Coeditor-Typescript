#!/usr/bin/env python3
"""
【guohx】简单分析Python版本的processed数据文件内容（不依赖numpy）
"""

import pickle
import sys
import os
from pathlib import Path

def simple_analyze():
    """【guohx】简单分析数据文件"""
    
    processed_dir = Path("../datasets_root/perm2k/processed/C3ProblemGenerator(VERSION=3.1, analyzer=())/")
    
    print("=" * 80)
    print("【guohx】Python版本 PROCESSED 数据简单分析")
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
                # 【guohx】直接读取pickle文件，不导入项目模块
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
                            print(f"    📝 类名: {first_problem.__class__.__name__}")
                            
                            # 【guohx】显示属性
                            if hasattr(first_problem, '__dict__'):
                                attrs = list(first_problem.__dict__.keys())
                                print(f"    📝 属性数量: {len(attrs)}")
                                print(f"    📝 属性列表: {attrs[:10]}{'...' if len(attrs) > 10 else ''}")
                                
                                # 【guohx】显示一些关键属性
                                key_attrs = ['repo_name', 'commit_hash', 'file_path', 'edit_type', 'pre_edit', 'post_edit']
                                for attr in key_attrs:
                                    if hasattr(first_problem, attr):
                                        attr_value = getattr(first_problem, attr)
                                        if isinstance(attr_value, str) and len(attr_value) > 100:
                                            attr_value = attr_value[:100] + "..."
                                        print(f"      {attr}: {attr_value}")
                
                elif isinstance(data, list):
                    print(f"📈 列表长度: {len(data)}")
                    if len(data) > 0:
                        first_item = data[0]
                        print(f"🔍 第一个元素类型: {type(first_item)}")
                        print(f"📝 类名: {first_item.__class__.__name__}")
                        if hasattr(first_item, '__dict__'):
                            attrs = list(first_item.__dict__.keys())
                            print(f"📝 属性数量: {len(attrs)}")
                            print(f"📝 属性列表: {attrs[:10]}{'...' if len(attrs) > 10 else ''}")
                
                print(f"\n✅ 总问题数量: {total_problems if 'total_problems' in locals() else 'N/A'}")
                
            except Exception as e:
                print(f"❌ 读取错误: {e}")
                # 【guohx】尝试更简单的读取方式
                try:
                    print("🔄 尝试简单读取...")
                    with open(file_path, 'rb') as f:
                        raw_data = f.read()
                    print(f"📊 原始数据大小: {len(raw_data)} bytes")
                    print(f"📊 前100字节: {raw_data[:100]}")
                except Exception as e2:
                    print(f"❌ 简单读取也失败: {e2}")
            
            print("=" * 80)

if __name__ == "__main__":
    simple_analyze()