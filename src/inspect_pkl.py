import pickle

# 修改为你想查看的文件路径
file_path = '../datasets_root/perm2k/transformed/eval-C3ProblemGenerator(VERSION=3.1, analyzer=())-C3ProblemChangeInlining(VERSION=1.4, max_inline_ratio=0.6)'

with open(file_path, 'rb') as f:
    data = pickle.load(f)
    print(f"Type: {type(data)}")
    if isinstance(data, list):
        print(f"Total items: {len(data)}")
        if data:
            print("First item example:")
            print(data[0])
    elif isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        for k, v in data.items():
            print(f"Key: {k}, Value type: {type(v)}, Example: {v[:1] if isinstance(v, list) else v}")
            break