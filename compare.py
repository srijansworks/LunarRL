"""Print evaluation summaries side by side, e.g. baseline vs your experiment.

    python compare.py ppo_baseline exp_fuel
"""
import json
import sys

from lunarrl.common import results_dir

KEYS = ["mean_return", "std_return", "median_return", "solved_episodes", "landed",
        "landed_on_pad", "crashed", "out_of_bounds", "timeout", "mean_length",
        "mean_main_engine_frames", "mean_side_engine_frames"]

runs = sys.argv[1:] or ["ppo_baseline"]
cols = []
for run in runs:
    s = json.loads((results_dir(run) / "eval_summary.json").read_text())
    c = json.loads((results_dir(run) / "train_config.json").read_text())
    cols.append((f"{run} (seed {c['seed']}, {c['reward_wrapper']})", s["ppo"]))
    if run == runs[0] and "random" in s:
        cols.insert(0, ("random policy", s["random"]))

width = max(len(n) for n, _ in cols) + 2
print("metric".ljust(16) + "".join(n.rjust(width) for n, _ in cols))
for k in KEYS:
    print(k.ljust(16) + "".join(str(s[k]).rjust(width) for _, s in cols))
print(f"\nepisodes per policy: {cols[-1][1]['episodes']}. Success = return >= 200.")
