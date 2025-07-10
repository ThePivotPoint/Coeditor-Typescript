"""
This script preprocesses the repos into the TSCommits format (TypeScript version).
You can safely skip this step since it will automatically be run when you
train a new model (and with the corresponding encoder parameters).

The raw repos will be loaded from `get_dataset_dir(dataset_name) / "repos"`, and the
processed results will be saved to `get_dataset_dir(dataset_name) / "processed"`
and `get_dataset_dir(dataset_name) / "transformed"`.
"""
import pickle
from coeditor._utils import run_long_task
from coeditor.common import *
from coeditor_ts.c3problem_ts import TSC3ProblemChangeInlining, TSC3ProblemGenerator
from coeditor_ts.dataset_ts import *

# === Add dataset interface for TS if missing ===
# from coeditor_ts.dataset_ts import make_or_load_ts_dataset, make_or_load_transformed_ts_dataset, TSC3CombinedEncoder

if __name__ == "__main__":
    os.chdir(proj_root())

    dataset_name = "perm2k_ts"
    encoder = TSC3CombinedEncoder(
        problem_tranform=TSC3ProblemChangeInlining(
            max_inline_ratio=0.6, allow_empty_problems=True
        ),
    )
    with run_long_task(
        f"Preparing dataset {dataset_name} with encoder {encoder.change_processor}"
    ):
        problems = make_or_load_ts_dataset(
            dataset_name,
            encoder.change_processor,
            # ("valid", "test", "train"),
            ("train",),
            remake_problems=False,
        )

        transformed = make_or_load_transformed_ts_dataset(
            dataset_name,
            problems,
            encoder,
        )

    tokenizer = TSC3ProblemTokenizer()
    for name, probs in transformed.items():
        print("=" * 40, name, "=" * 40)
        stats = tokenizer.compute_stats(probs)
        from coeditor._utils import pretty_print_dict
        pretty_print_dict(stats) 