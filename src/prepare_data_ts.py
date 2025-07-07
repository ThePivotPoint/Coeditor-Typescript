#!/usr/bin/env python3
"""
TypeScript 版本的数据预处理脚本
基于原始的 prepare_data.py，但处理 TypeScript 文件而不是 Python 文件
"""
import pickle
from coeditor.common import *
from coeditor_ts.ts_dataset import TypeScriptDatasetProcessor
import argparse
import os
from typing import Sequence, cast

if __name__ == "__main__":
    os.chdir(proj_root())

    dataset_name = "perm2k_ts"
    # 直接创建 TypeScript 数据集处理器
    processor = TypeScriptDatasetProcessor(
        dataset_dir=get_dataset_dir(dataset_name),
        max_workers=4
    )
    print(f"开始处理 TypeScript 数据集: {dataset_name}")
    processor.process_full_dataset()
    print("TypeScript 数据处理完成!") 