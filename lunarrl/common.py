"""Environment factory, outcome classification and seeding helpers."""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import gymnasium as gym
import numpy as np

ENV_ID = "LunarLander-v3"
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"

# Gymnasium's own "solved" bar for LunarLander: an episode return of 200 or more.
SOLVED_RETURN = 200.0
# The pad sits between the two flags at x = -0.2 .. 0.2 in the observation's units.
PAD_HALF_WIDTH = 0.2


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch. Box2D itself is seeded through env.reset(seed=...)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # random baseline does not need torch
        pass


def make_env(reward_wrapper: str = "none", render_mode: str | None = None) -> gym.Env:
    env = gym.make(ENV_ID, render_mode=render_mode)
    if reward_wrapper != "none":
        from .wrappers import WRAPPERS

        env = WRAPPERS[reward_wrapper](env)
    return env


@dataclass
class EpisodeResult:
    policy: str
    seed: int
    episode_return: float  # always the ORIGINAL Gymnasium reward, never a wrapped one
    length: int
    outcome: str  # landed | crashed | out_of_bounds | timeout
    on_pad: bool
    solved: bool
    main_engine_frames: int = 0  # frames with action 2 (Gymnasium charges 0.3 each)
    side_engine_frames: int = 0  # frames with action 1 or 3 (0.03 each)


def classify(env: gym.Env, terminated: bool, truncated: bool, obs: np.ndarray) -> tuple[str, bool]:
    """Work out why an episode ended, using the same flags Gymnasium uses internally.

    LunarLander ends with reward -100 when the body touches the ground (game_over)
    or the lander leaves the screen (|x| >= 1), and with +100 when the lander
    comes to rest (Box2D puts the body to sleep). TimeLimit truncates at 1000 steps.
    """
    u = env.unwrapped
    x = float(obs[0])
    if truncated and not terminated:
        return "timeout", False
    if u.game_over:
        return "crashed", False
    if abs(x) >= 1.0:
        return "out_of_bounds", False
    # remaining case: the lander body went to sleep, i.e. it landed and stopped
    both_legs = bool(obs[6] > 0.5 and obs[7] > 0.5)
    return "landed", both_legs and abs(x) <= PAD_HALF_WIDTH


def run_episode(env: gym.Env, act, seed: int, policy_name: str, frames: list | None = None) -> EpisodeResult:
    """Play one episode. `act(obs) -> int`. Returns the unwrapped return."""
    obs, _ = env.reset(seed=seed)
    total, steps, main, side = 0.0, 0, 0, 0
    while True:
        if frames is not None:
            frames.append(env.render())
        action = act(obs)
        main += action == 2
        side += action in (1, 3)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(info.get("original_reward", reward))
        steps += 1
        if terminated or truncated:
            break
    if frames is not None:
        frames.append(env.render())
    outcome, on_pad = classify(env, terminated, truncated, obs)
    return EpisodeResult(policy_name, seed, round(total, 3), steps, outcome, on_pad,
                         total >= SOLVED_RETURN, int(main), int(side))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def results_dir(run_name: str) -> Path:
    # The published run lives in results/<run>; your own experiments can go anywhere.
    return Path(os.environ.get("LUNARRL_RESULTS", RESULTS_DIR)) / run_name


__all__ = ["EpisodeResult", "asdict"]
