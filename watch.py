"""Watch an agent land in a window (needs a display; uses pygame).

    python watch.py                      # trained model, first evaluation seed
    python watch.py --random --seed 3    # random policy for comparison
"""
import argparse

import numpy as np

from lunarrl.common import MODELS_DIR, make_env, run_episode

p = argparse.ArgumentParser()
p.add_argument("--run-name", default="ppo_baseline")
p.add_argument("--seed", type=int, default=10_000)
p.add_argument("--random", action="store_true")
a = p.parse_args()

if a.random:
    rng = np.random.default_rng(a.seed)
    act = lambda o: int(rng.integers(4))
else:
    from stable_baselines3 import PPO

    model = PPO.load(MODELS_DIR / a.run_name / "model.zip", device="cpu")
    act = lambda o: int(model.predict(o, deterministic=True)[0])

env = make_env(render_mode="human")
r = run_episode(env, act, a.seed, "random" if a.random else "ppo")
env.close()
print(f"return {r.episode_return:.1f}, {r.length} steps, outcome: {r.outcome}, on pad: {r.on_pad}")
