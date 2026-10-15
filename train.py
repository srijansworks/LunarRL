"""Train PPO on Gymnasium LunarLander-v3.

    python train.py                                  # the published run (1M steps)
    python train.py --timesteps 100000 --run-name quick
    python train.py --run-name fuel --reward-wrapper fuel   # experiment

Writes models/<run>/model.zip, runs/<run>/ (raw Monitor logs) and
results/<run>/training_episodes.csv + train_config.json.
"""
from __future__ import annotations

import argparse
import csv
import platform
import time

import gymnasium
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from lunarrl.common import ENV_ID, MODELS_DIR, RUNS_DIR, make_env, results_dir, seed_everything, write_json
from lunarrl.wrappers import WRAPPERS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default="ppo_baseline")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--reward-wrapper", choices=["none", *WRAPPERS], default="none")
    # PPO hyperparameters. Defaults follow rl-baselines3-zoo's tuned LunarLander PPO config.
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--gae-lambda", type=float, default=0.98)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--clip-range", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    seed_everything(a.seed)
    torch.set_num_threads(1)  # small MLP: one thread is faster and more repeatable

    log_dir = RUNS_DIR / a.run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    def factory(rank: int):
        def _init():
            # Monitor sits BELOW any reward wrapper, so logged returns are the original reward.
            env = make_env()
            env = Monitor(env, str(log_dir / f"env{rank:02d}"))
            if a.reward_wrapper != "none":
                env = WRAPPERS[a.reward_wrapper](env)
            return env
        return _init

    venv = DummyVecEnv([factory(i) for i in range(a.n_envs)])
    venv.seed(a.seed)  # env i gets seed a.seed + i on its first reset

    model = PPO(
        "MlpPolicy", venv,
        learning_rate=a.learning_rate, n_steps=a.n_steps, batch_size=a.batch_size,
        n_epochs=a.n_epochs, gamma=a.gamma, gae_lambda=a.gae_lambda,
        ent_coef=a.ent_coef, clip_range=a.clip_range,
        seed=a.seed, device="cpu", verbose=1,
    )
    t0 = time.time()
    model.learn(total_timesteps=a.timesteps, progress_bar=False)
    wall = time.time() - t0

    MODELS_DIR.joinpath(a.run_name).mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / a.run_name / "model.zip")
    venv.close()

    out = results_dir(a.run_name)
    n_eps = merge_monitor_logs(log_dir, out / "training_episodes.csv")
    write_json(out / "train_config.json", {
        "env_id": ENV_ID, "algorithm": "PPO", "policy": "MlpPolicy (2x64 tanh, SB3 default)",
        "run_name": a.run_name, "seed": a.seed, "n_envs": a.n_envs,
        "total_timesteps": int(model.num_timesteps), "reward_wrapper": a.reward_wrapper,
        "hyperparameters": {k: getattr(a, k) for k in (
            "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "ent_coef", "clip_range")},
        "training_episodes": n_eps, "wall_clock_seconds": round(wall, 1),
        "versions": {"python": platform.python_version(), "gymnasium": gymnasium.__version__,
                     "stable_baselines3": stable_baselines3.__version__, "torch": torch.__version__},
        "machine": f"{platform.system()} {platform.machine()}, CPU only",
    })
    print(f"\nSaved model to models/{a.run_name}/model.zip after {model.num_timesteps} steps in {wall/60:.1f} min")
    print(f"Training history: {out/'training_episodes.csv'} ({n_eps} episodes)")


def merge_monitor_logs(log_dir, dest) -> int:
    """Combine the per-env Monitor CSVs into one history.

    DummyVecEnv steps all envs in lock-step, so an episode that ends after an env's
    k-th step ends at global step k * n_envs. Ordering by that (not by wall-clock
    time) makes the file identical across reruns with the same seed.
    """
    files = sorted(log_dir.glob("env*.monitor.csv"))
    rows = []
    for idx, f in enumerate(files):
        with f.open() as fh:
            next(fh)  # Monitor's JSON header line
            env_steps = 0
            for r in csv.DictReader(fh):
                env_steps += int(r["l"])
                rows.append((env_steps * len(files), idx, float(r["r"]), int(r["l"])))
    rows.sort()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["episode", "timesteps_end", "episode_return", "length"])
        for i, (step, _, ret, length) in enumerate(rows, 1):
            w.writerow([i, step, round(ret, 3), length])
    return len(rows)


if __name__ == "__main__":
    main()
