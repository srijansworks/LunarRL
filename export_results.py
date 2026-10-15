"""Turn the outputs of train.py and evaluate.py into files the portfolio can serve.

    python export_results.py                      # uses results/ppo_baseline
    python export_results.py --extra-runs ppo_seed1 ppo_seed2 --experiment exp_fuel

Writes, inside lunar-rl/results/<run>/:
  replay_ppo.mp4, replay_random.mp4 (+ .jpg posters)  one episode each, same seed
  site_data.json                                     everything the project page plots
and, when --site-dir points at the portfolio root (default: the folder above
lunar-rl/ if it contains index.html), copies them to:
  assets/media/lunar-rl-*.mp4|jpg and assets/data/lunarrl-results.js
No numbers are typed in by hand: every value on the page comes from these files.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from lunarrl.common import MODELS_DIR, ROOT, asdict, make_env, results_dir, run_episode


def training_curve(csv_path: Path, bins: int = 60) -> list[dict]:
    """Mean / 10th / 90th percentile of episode return in equal-width timestep bins."""
    rows = list(csv.DictReader(csv_path.open()))
    steps = np.array([int(r["timesteps_end"]) for r in rows])
    rets = np.array([float(r["episode_return"]) for r in rows])
    edges = np.linspace(0, steps.max(), bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (steps > lo) & (steps <= hi)
        if m.sum() == 0:
            continue
        r = rets[m]
        out.append({"step": int(hi), "mean": round(float(r.mean()), 1),
                    "p10": round(float(np.percentile(r, 10)), 1),
                    "p90": round(float(np.percentile(r, 90)), 1), "n": int(m.sum())})
    return out


def record(run: str, seed: int, out: Path, size=(480, 320)) -> dict:
    """Record one trained and one random episode on the same seed as MP4 + poster."""
    from stable_baselines3 import PPO

    model = PPO.load(MODELS_DIR / run / "model.zip", device="cpu")
    rng = np.random.default_rng(seed)
    policies = {"ppo": lambda o: int(model.predict(o, deterministic=True)[0]),
                "random": lambda o: int(rng.integers(4))}
    meta = {}
    for name, act in policies.items():
        env = make_env(render_mode="rgb_array")
        frames: list = []
        res = run_episode(env, act, seed, name, frames)
        env.close()
        mp4 = out / f"replay_{name}.mp4"
        # yuv420p + no audio track: plays everywhere, nothing can autoplay with sound
        w = imageio.get_writer(mp4, fps=50, codec="libx264", quality=None, macro_block_size=1,
                               pixelformat="yuv420p", output_params=["-an", "-movflags", "+faststart"],
                               ffmpeg_params=["-crf", "28", "-vf", f"scale={size[0]}:{size[1]}"])
        # the final frame is held for 1 s so the ending is visible; the episode itself is unchanged
        for f in frames + [frames[-1]] * 50:
            w.append_data(f)
        w.close()
        # poster: a frame about two thirds through, where the descent is visible
        imageio.imwrite(out / f"replay_{name}.jpg", frames[int(len(frames) * 0.66)], quality=85)
        meta[name] = asdict(res) | {"frames": len(frames), "seconds": round(len(frames) / 50, 1)}
        print(f"recorded {name}: return {res.episode_return:.1f}, {res.outcome}, {len(frames)} frames")
    return meta


def load(run: str, name: str) -> dict:
    return json.loads((results_dir(run) / name).read_text())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default="ppo_baseline")
    p.add_argument("--extra-runs", nargs="*", default=[], help="other seeds of the same config")
    p.add_argument("--experiment", default="", help="a run trained with a changed hyperparameter or wrapper")
    p.add_argument("--replay-seed", type=int, default=None,
                   help="defaults to the FIRST evaluation seed, so the clip is not hand-picked")
    p.add_argument("--site-dir", default=str(ROOT.parent))
    p.add_argument("--no-video", action="store_true")
    a = p.parse_args()

    out = results_dir(a.run_name)
    summary = load(a.run_name, "eval_summary.json")
    config = load(a.run_name, "train_config.json")
    episodes = list(csv.DictReader((out / "eval_episodes.csv").open()))
    replay_seed = a.replay_seed if a.replay_seed is not None else int(summary["seeds"].split("..")[0])

    data = {
        "generated_by": "lunar-rl/export_results.py",
        "run": a.run_name, "config": config, "eval": summary,
        "eval_returns": {pol: [float(e["episode_return"]) for e in episodes if e["policy"] == pol]
                         for pol in ("ppo", "random")},
        "training_curve": training_curve(out / "training_episodes.csv"),
        "other_seeds": [{"run": r, "seed": load(r, "train_config.json")["seed"],
                         "ppo": load(r, "eval_summary.json")["ppo"]} for r in a.extra_runs],
    }
    if a.experiment:
        data["experiment"] = {"run": a.experiment, "config": load(a.experiment, "train_config.json"),
                              "ppo": load(a.experiment, "eval_summary.json")["ppo"]}
    if not a.no_video:
        data["replay"] = {"seed": replay_seed, **record(a.run_name, replay_seed, out)}

    (out / "site_data.json").write_text(json.dumps(data, indent=1) + "\n")
    print(f"Wrote {out/'site_data.json'}")

    site = Path(a.site_dir)
    if (site / "index.html").exists():
        (site / "assets/data").mkdir(parents=True, exist_ok=True)
        (site / "assets/data/lunarrl-results.js").write_text(
            "/* Generated by lunar-rl/export_results.py from real runs. Do not edit by hand. */\n"
            "window.LUNARRL_RESULTS = " + json.dumps(data, separators=(",", ":")) + ";\n")
        if not a.no_video:
            for name in ("ppo", "random"):
                for ext in ("mp4", "jpg"):
                    shutil.copy(out / f"replay_{name}.{ext}", site / f"assets/media/lunar-rl-{name}.{ext}")
        print(f"Copied site data and media into {site}")


if __name__ == "__main__":
    main()
