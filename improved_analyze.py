#!/usr/bin/env python3
"""
【guohx】改进的分析脚本，能够处理依赖coeditor模块的pickle文件
"""

import pickle
import sys
import os
from pathlib import Path

# 【guohx】添加项目路径，确保能找到coeditor模块
sys.path.insert(0, 'src')

def improved_analyze():
    """【guohx】改进的数据分析"""
    
    # 【guohx】设置要分析的目录路径
    processed_dir = Path("datasets_root/perm2k/processed/C3ProblemGenerator(VERSION=3.1, analyzer=())/")
    
    print("=" * 80)
    print("【guohx】改进的 PROCESSED 数据分析")
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
                # 【guohx】尝试导入coeditor模块
                try:
                    import coeditor
                    print("✅ coeditor模块导入成功")
                except ImportError as e:
                    print(f"⚠️ coeditor模块导入失败: {e}")
                    print("🔄 尝试直接读取pickle...")
                
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
                            print(f"    📝 类名: {first_problem.__class__.__name__}")
                            
                            # 【guohx】显示属性
                            if hasattr(first_problem, '__dict__'):
                                attrs = list(first_problem.__dict__.keys())
                                print(f"    📝 属性数量: {len(attrs)}")
                                print(f"    📝 属性列表: {attrs}")
                                
                                # 【guohx】显示一些关键属性
                                key_attrs = ['repo_name', 'commit_hash', 'file_path', 'edit_type', 'pre_edit', 'post_edit', 'span', 'context']
                                for attr in key_attrs:
                                    if hasattr(first_problem, attr):
                                        attr_value = getattr(first_problem, attr)
                                        if isinstance(attr_value, str) and len(attr_value) > 100:
                                            attr_value = attr_value[:100] + "..."
                                        print(f"      {attr}: {attr_value}")
                                
                                # 【guohx】显示所有属性的详细信息
                                print(f"\n    📝 所有属性详细信息:")
                                for attr_name, attr_value in first_problem.__dict__.items():
                                    attr_type = type(attr_value).__name__
                                    if isinstance(attr_value, str):
                                        if len(attr_value) > 50:
                                            display_value = attr_value[:50] + "..."
                                        else:
                                            display_value = attr_value
                                    elif hasattr(attr_value, '__len__'):
                                        display_value = f"[长度: {len(attr_value)}]"
                                    else:
                                        display_value = str(attr_value)
                                    print(f"      {attr_name} ({attr_type}): {display_value}")
                
                elif isinstance(data, list):
                    print(f"📈 列表长度: {len(data)}")
                    if len(data) > 0:
                        first_item = data[0]
                        print(f"🔍 第一个元素类型: {type(first_item)}")
                        print(f"📝 类名: {first_item.__class__.__name__}")
                        if hasattr(first_item, '__dict__'):
                            attrs = list(first_item.__dict__.keys())
                            print(f"📝 属性数量: {len(attrs)}")
                            print(f"📝 属性列表: {attrs}")
                
                print(f"\n✅ 总问题数量: {total_problems if 'total_problems' in locals() else 'N/A'}")
                
            except Exception as e:
                print(f"❌ 读取错误: {e}")
                import traceback
                traceback.print_exc()
                
                # 【guohx】尝试更简单的读取方式
                try:
                    print("🔄 尝试简单读取...")
                    with open(file_path, 'rb') as f:
                        raw_data = f.read()
                    print(f"📊 原始数据大小: {len(raw_data)} bytes")
                    print(f"📊 前200字节: {raw_data[:200]}")
                except Exception as e2:
                    print(f"❌ 简单读取也失败: {e2}")
            
            print("=" * 80)

def analyze_specific_file(file_path_str):
    """【guohx】分析指定的单个文件"""
    
    file_path = Path(file_path_str)
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return
    
    print(f"\n🔍 分析指定文件: {file_path}")
    print("=" * 80)
    
    try:
        # 【guohx】尝试导入coeditor模块
        try:
            import coeditor
            print("✅ coeditor模块导入成功")
        except ImportError as e:
            print(f"⚠️ coeditor模块导入失败: {e}")
        
        # 【guohx】读取pickle文件
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"📋 数据类型: {type(data)}")
        
        if isinstance(data, dict):
            print(f"🔑 字典键: {list(data.keys())}")
            
            for key, value in data.items():
                print(f"\n📂 分集: {key}")
                print(f"📊 类型: {type(value)}")
                print(f"📈 长度: {len(value) if hasattr(value, '__len__') else 'N/A'}")
                
                if hasattr(value, '__len__') and len(value) > 0:
                    print(f"🔍 第一个问题类型: {type(value[0])}")
                    print(f"📝 类名: {value[0].__class__.__name__}")
                    
                    if hasattr(value[0], '__dict__'):
                        attrs = list(value[0].__dict__.keys())
                        print(f"📝 属性数量: {len(attrs)}")
                        print(f"📝 属性列表: {attrs}")
        
        elif isinstance(data, list):
            print(f"📈 列表长度: {len(data)}")
            if len(data) > 0:
                print(f"🔍 第一个元素类型: {type(data[0])}")
                print(f"📝 类名: {data[0].__class__.__name__}")
                
    except Exception as e:
        print(f"❌ 读取错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 【guohx】分析所有文件
    improved_analyze()
    
    # 【guohx】分析特定文件（可选）
    # analyze_specific_file("datasets_root/perm2k/processed/C3ProblemGenerator(VERSION=3.1, analyzer=())/deepseek-ai~DeepSeek-V3(1000, is_training=False)") 