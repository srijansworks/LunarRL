"""Evaluate a trained PPO model and a uniform-random policy on the same seeds.

    python evaluate.py                        # models/ppo_baseline, 100 episodes each
    python evaluate.py --run-name fuel --episodes 50

Both policies play the SAME list of seeds (--seed-start .. +episodes), which are
different from the training seeds. Writes results/<run>/eval_episodes.csv and
eval_summary.json.
"""
from __future__ import annotations

import argparse
import csv
import statistics

import numpy as np

from lunarrl.common import ENV_ID, MODELS_DIR, SOLVED_RETURN, EpisodeResult, asdict, make_env, results_dir, run_episode, seed_everything, write_json


def summarize(rows: list[EpisodeResult]) -> dict:
    rets = [r.episode_return for r in rows]
    count = lambda pred: sum(1 for r in rows if pred(r))
    return {
        "episodes": len(rows),
        "mean_return": round(statistics.fmean(rets), 2),
        "std_return": round(statistics.pstdev(rets), 2),
        "median_return": round(statistics.median(rets), 2),
        "min_return": round(min(rets), 2),
        "max_return": round(max(rets), 2),
        "solved_episodes": count(lambda r: r.solved),
        "landed": count(lambda r: r.outcome == "landed"),
        "landed_on_pad": count(lambda r: r.on_pad),
        "crashed": count(lambda r: r.outcome == "crashed"),
        "out_of_bounds": count(lambda r: r.outcome == "out_of_bounds"),
        "timeout": count(lambda r: r.outcome == "timeout"),
        "mean_length": round(statistics.fmean(r.length for r in rows), 1),
        "mean_main_engine_frames": round(statistics.fmean(r.main_engine_frames for r in rows), 1),
        "mean_side_engine_frames": round(statistics.fmean(r.side_engine_frames for r in rows), 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default="ppo_baseline")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed-start", type=int, default=10_000)
    p.add_argument("--skip-random", action="store_true")
    a = p.parse_args()

    from stable_baselines3 import PPO

    seeds = list(range(a.seed_start, a.seed_start + a.episodes))
    model = PPO.load(MODELS_DIR / a.run_name / "model.zip", device="cpu")
    env = make_env()  # evaluation always uses the unmodified environment

    trained = [run_episode(env, lambda o: int(model.predict(o, deterministic=True)[0]), s, "ppo") for s in seeds]
    rows = list(trained)
    summary = {"env_id": ENV_ID, "run_name": a.run_name, "seeds": f"{seeds[0]}..{seeds[-1]}",
               "success_definition": f"episode return >= {SOLVED_RETURN:g} (Gymnasium's 'solved' threshold)",
               "ppo": summarize(trained)}

    if not a.skip_random:
        seed_everything(a.seed_start)
        rng = np.random.default_rng(a.seed_start)
        rand = [run_episode(env, lambda o: int(rng.integers(4)), s, "random") for s in seeds]
        rows += rand
        summary["random"] = summarize(rand)
    env.close()

    out = results_dir(a.run_name)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "eval_episodes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(rows[0])))
        w.writeheader()
        w.writerows(asdict(r) for r in rows)
    write_json(out / "eval_summary.json", summary)

    for name in ("ppo", "random"):
        if name in summary:
            s = summary[name]
            print(f"{name:>6}: mean {s['mean_return']:8.1f} ± {s['std_return']:6.1f} | "
                  f"solved {s['solved_episodes']}/{s['episodes']} | landed {s['landed']} "
                  f"(on pad {s['landed_on_pad']}) | crashed {s['crashed']} | "
                  f"out of bounds {s['out_of_bounds']} | timeout {s['timeout']}")
    print(f"Wrote {out/'eval_summary.json'}")


if __name__ == "__main__":
    main()
