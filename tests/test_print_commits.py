from pathlib import Path
from coeditor.git import get_commit_history

if __name__ == "__main__":
    root = Path("datasets_root/perm2k_ts/repos/test/pmndrs~zustand")
    max_history_per_repo = 10
    commits = get_commit_history(root)[-max_history_per_repo:]
    print(f"Total commits: {len(commits)}")
    print("--- Commits ---")
    for c in commits:
        print(c.summary())
        print(f"  hash: {c.hash}")
        print(f"  parents: {c.parents}")
        print(f"  msg: {c.msg}")
        print() 