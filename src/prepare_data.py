"""
【guohx】本脚本用于将原始代码仓库预处理为 PyCommits 格式的数据集。
【guohx】PyCommits 格式用于论文中的代码自动编辑任务。
【guohx】通常无需手动运行本脚本，在训练新模型时会自动调用。
【guohx】原始仓库数据将从 `get_dataset_dir(dataset_name) / "repos"` 加载，
【guohx】处理后的结果将保存到 `get_dataset_dir(dataset_name) / "processed"` 和 `get_dataset_dir(dataset_name) / "transformed"`。
"""
# 【guohx】导入标准库和项目内工具
import pickle
from coeditor._utils import run_long_task
from coeditor.c3problem import C3ProblemChangeInlining, C3ProblemGenerator
from coeditor.common import *
from coeditor.dataset import *

# 【guohx】主程序入口，数据处理主流程
if __name__ == "__main__":
    # 【guohx】切换到项目根目录，确保后续所有相对路径操作都正确
    os.chdir(proj_root())

    # 【guohx】设置要处理的数据集名称（如 perm2k_ts），并初始化 TypeScript 相关的编码器、问题变换器等
    # 【guohx】此处需将所有 Python 相关的类、参数替换为 TS 版本
    dataset_name = "perm2k"
    encoder = C3CombinedEncoder(
        problem_tranform=C3ProblemChangeInlining(
            max_inline_ratio=0.6, allow_empty_problems=True
        ),
    )
    
    # 【guohx】使用 run_long_task 包裹，显示进度和耗时
    with run_long_task(
        f"Preparing dataset {dataset_name} with encoder {encoder.change_processor}"
    ):
        # 【guohx】生成/加载结构化编辑问题数据集（如 TSC3Problem），只处理 train 分集
        # 【guohx】调用 TS 版本的 make_or_load_dataset，批量处理原始 TS 仓库，生成结构化的“编辑问题”对象
        # 【guohx】此处需确保 change_processor、数据集路径等均为 TS 相关实现
        problems = make_or_load_dataset(
            dataset_name,
            encoder.change_processor,
            # 可选分集：("valid", "test", "train")
            ("train",),
            remake_problems=False,  # 是否强制重新生成问题
        )


        # 【guohx】生成/加载变换后的数据集，对上一步生成的“编辑问题”对象做进一步变换、增强、编码，得到最终可用于模型训练/评估的数据
        # 【guohx】需用 TS 相关的 encoder、tokenizer 等
        transformed = make_or_load_transformed_dataset(
            dataset_name,
            problems,
            encoder,
        )
        # 【guohx】make_or_load_transformed_dataset 细节：
        #   - 对每个 TSC3Problem 做数据增强、变换（如上下文扩展、负样本生成等）
        #   - 将增强后的问题对象编码为模型可直接使用的输入格式（如 token 序列）
        #   - 将变换/编码后的数据缓存/保存，便于后续训练和复用

    # 【guohx】初始化 TypeScript 版本的 Tokenizer，用于后续 token 统计、分析等
    tokenizer = C3ProblemTokenizer()
    # 【guohx】遍历每个数据分集（如 train/valid/test），用 TS Tokenizer 统计 token 分布、长度等信息，并输出结果
    for name, probs in transformed.items():
        # 类型转换，确保 probs 是 C3Problem 的序列
        probs = cast(Sequence[C3Problem], probs)
        print("=" * 40, name, "=" * 40)
        # 【guohx】Token统计：
        #   - 遍历每个数据分集，分别统计 token 信息
        #   - 用 TS 版本的 Tokenizer 统计 token 分布、长度、参考上下文等详细信息
        #   - 以表格或日志形式输出统计结果，便于分析数据集特征
        stats = tokenizer.compute_stats(probs)
        pretty_print_dict(stats)
