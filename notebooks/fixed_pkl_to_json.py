#!/usr/bin/env python3
"""
【guohx】修复版本的pkl_to_json脚本，适配你的数据结构
"""

import sys
import pickle
import json
from dataclasses import is_dataclass, fields
from pathlib import Path

# 【guohx】添加项目路径
sys.path.insert(0, "/Users/feiyu/Desktop/code/NewCoEditor/src")

try:
    from coeditor.common import *
    from coeditor.dataset import *
    print("✅ coeditor模块导入成功")
except ImportError as e:
    print(f"⚠️ coeditor模块导入失败: {e}")
    print("🔄 继续执行，但可能无法处理某些特殊类型")

def instance_to_json(obj: Any) -> Dict[str, Any]:
    """
    【guohx】将dataclass实例或dict实例序列化为JSON可序列化的字典
    """
    if not is_dataclass(obj) and not isinstance(obj, dict):
        raise TypeError(f"Expected a dataclass instance, got {type(obj).__name__}")
    
    def _serialize(value: Any) -> Any:
        if is_dataclass(value):
            result = {"__class__": type(value).__name__}

            for field in fields(value):
                field_value = getattr(value, field.name)
                result[field.name] = _serialize(field_value)

            return result
        elif isinstance(value, dict):
            return {k: _serialize(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple, set)):
            return [_serialize(item) for item in value]
        # 【guohx】处理NumPy数组
        elif hasattr(value, 'tolist'):  # 兼容numpy数组
            return value.tolist()
        # 【guohx】处理range对象
        elif isinstance(value, range):
            return {
                "__type__": "range",
                "start": value.start,
                "stop": value.stop,
                "step": value.step
            }
        else:
            return value
    
    return _serialize(obj)

def serialize_to_json(obj: Any, fp: Optional[str | Path] = None, indent: int = 2) -> Optional[str]:
    """
    【guohx】将对象序列化为JSON字符串或写入文件
    """
    data = instance_to_json(obj)
    
    if fp is not None:
        with open(fp, 'w') as f:
            json.dump(data, f, indent=indent)
        return None
    
    return json.dumps(data, indent=indent)

def analyze_and_convert():
    """【guohx】分析并转换你的数据文件"""
    
    # 【guohx】设置文件路径
    pickle_abs_path = Path("/Users/feiyu/Desktop/code/NewCoEditor/datasets_root/perm2k/processed/C3ProblemGenerator(VERSION=3.1, analyzer=())/deepseek-ai~DeepSeek-V3(1000, is_training=False)")
    json_abs_path = Path("/Users/feiyu/Desktop/code/NewCoEditor/datasets_root/perm2k/")
    
    print(f"📁 读取文件: {pickle_abs_path}")
    print(f"📁 输出目录: {json_abs_path}")
    
    if not pickle_abs_path.exists():
        print(f"❌ 文件不存在: {pickle_abs_path}")
        return
    
    try:
        # 【guohx】读取pickle文件
        with open(pickle_abs_path, "rb") as f:
            problems = pickle.load(f)
        
        print(f"📋 数据类型: {type(problems)}")
        
        if isinstance(problems, dict):
            print(f"🔑 字典键: {list(problems.keys())}")
            
            # 【guohx】遍历每个分集
            for split_name, split_data in problems.items():
                print(f"\n📂 分集: {split_name}")
                print(f"📊 类型: {type(split_data)}")
                print(f"📈 长度: {len(split_data) if hasattr(split_data, '__len__') else 'N/A'}")
                
                if hasattr(split_data, '__len__') and len(split_data) > 0:
                    # 【guohx】转换第一个问题为JSON
                    first_problem = split_data[0]
                    print(f"🔍 第一个问题类型: {type(first_problem)}")
                    print(f"📝 类名: {first_problem.__class__.__name__}")
                    
                    # 【guohx】保存第一个问题
                    output_file = json_abs_path / f"sample_problem_{split_name}_0.json"
                    serialize_to_json(first_problem, output_file)
                    print(f"✅ 已保存到: {output_file}")
                    
                    # 【guohx】保存整个分集（可选，可能很大）
                    if len(split_data) <= 10:  # 只保存小数据集
                        output_file = json_abs_path / f"problems_{split_name}.json"
                        serialize_to_json(split_data, output_file)
                        print(f"✅ 已保存整个分集到: {output_file}")
                    else:
                        print(f"⚠️ 分集太大({len(split_data)}个问题)，跳过保存整个分集")
        
        elif isinstance(problems, list):
            print(f"📈 列表长度: {len(problems)}")
            if len(problems) > 0:
                first_problem = problems[0]
                print(f"🔍 第一个问题类型: {type(first_problem)}")
                print(f"📝 类名: {first_problem.__class__.__name__}")
                
                # 【guohx】保存第一个问题
                output_file = json_abs_path / "sample_problem_0.json"
                serialize_to_json(first_problem, output_file)
                print(f"✅ 已保存到: {output_file}")
        
        print("\n🎉 分析完成！")
        
    except Exception as e:
        print(f"❌ 处理错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_and_convert() 