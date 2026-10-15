"""Fast checks (about 5 s): python -m unittest discover -s tests"""
import unittest

import numpy as np

from lunarrl.common import make_env, run_episode
from lunarrl.wrappers import FuelPenalty

OUTCOMES = {"landed", "crashed", "out_of_bounds", "timeout"}


def random_policy(seed):
    rng = np.random.default_rng(seed)
    return lambda obs: int(rng.integers(4))


class TestLunarRL(unittest.TestCase):
    def test_spaces(self):
        env = make_env()
        self.assertEqual(env.observation_space.shape, (8,))
        self.assertEqual(env.action_space.n, 4)

    def test_random_episodes_end_with_known_outcome(self):
        env = make_env()
        for seed in range(5):
            r = run_episode(env, random_policy(seed), seed, "random")
            self.assertIn(r.outcome, OUTCOMES)
            self.assertEqual(r.main_engine_frames + r.side_engine_frames <= r.length, True)

    def test_same_seed_is_reproducible(self):
        a = run_episode(make_env(), random_policy(1), 123, "random")
        b = run_episode(make_env(), random_policy(1), 123, "random")
        self.assertEqual(a, b)

    def test_wrappers_report_original_return(self):
        base = run_episode(make_env(), random_policy(2), 7, "random")
        for name in ("fuel", "softcrash"):
            wrapped = run_episode(make_env(name), random_policy(2), 7, "random")
            self.assertAlmostEqual(base.episode_return, wrapped.episode_return, places=6)

    def test_fuel_penalty_lowers_learning_reward(self):
        env = FuelPenalty(make_env(reward_wrapper="none"))
        env.reset(seed=0)
        _, r, _, _, info = env.step(2)
        self.assertAlmostEqual(r, info["original_reward"] - 0.3)


if __name__ == "__main__":
    unittest.main()
