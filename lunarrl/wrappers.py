"""Optional reward wrappers for the experiment in LEARN-LUNARRL.md.

A wrapper only changes the reward the AGENT learns from. Evaluation always
reports the original Gymnasium return (passed through as info["original_reward"]),
so runs trained with different wrappers stay comparable.
"""
from __future__ import annotations

import gymnasium as gym


class FuelPenalty(gym.Wrapper):
    """Extra cost per engine firing, on top of Gymnasium's own fuel cost.

    Gymnasium already charges 0.3 per frame of main engine and 0.03 per frame of
    a side engine. This adds `scale` times that again, nudging the agent to land
    with less thrust. Actions: 0 noop, 1 left engine, 2 main engine, 3 right engine.
    """

    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["original_reward"] = reward
        extra = 0.3 if action == 2 else 0.03 if action in (1, 3) else 0.0
        return obs, reward - self.scale * extra, terminated, truncated, info


class SoftCrash(gym.Wrapper):
    """Halves the -100 crash/out-of-bounds penalty to -50. Expect riskier landings."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["original_reward"] = reward
        if terminated and reward == -100:
            reward = -50.0
        return obs, reward, terminated, truncated, info


WRAPPERS = {"fuel": FuelPenalty, "softcrash": SoftCrash}
