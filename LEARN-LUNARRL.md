# Learn LunarRL in under an hour

Written for Srijan. Work through it in order in a terminal inside `lunar-rl/` with the
virtual environment active (`source .venv/bin/activate`; setup is in `README.md`).
Times are rough. Keep a notes file open and write down every number you see; part 6
asks you to explain them.

---

## 1. Run one short training session (10 min)

```bash
python train.py --run-name learn_short --timesteps 100000
```

PPO prints a table every 16,384 steps (16 environments × 1,024 steps each). Watch four lines:

| Line | What it means | What you want to see |
|---|---|---|
| `rollout/ep_rew_mean` | mean return of the last 100 finished training episodes | climbing from about -200 |
| `rollout/ep_len_mean` | how long episodes last | short at first (fast crashes), then longer as it learns to hover, then shorter again once it lands |
| `train/entropy_loss` | minus the policy's randomness | moving toward 0 as the policy gets more confident |
| `train/clip_fraction` | share of updates PPO had to clip | small (≈0.05–0.2); large means updates are too aggressive |

Then score it on fixed seeds:

```bash
python evaluate.py --run-name learn_short --episodes 30
```

100k steps is about a tenth of the published run. Expect it to beat random but rarely
reach 200. Write down its mean return and crash count.

## 2. Read the observation and action spaces (5 min)

```bash
python - <<'PY'
import gymnasium as gym
env = gym.make("LunarLander-v3")
print(env.observation_space)   # 8 numbers
print(env.action_space)        # Discrete(4)
names = ["x", "y", "vx", "vy", "angle", "ang_vel", "left_leg", "right_leg"]
acts = ["noop", "left engine", "MAIN engine", "right engine"]
obs, _ = env.reset(seed=1)
for t in range(5):
    a = 2                                    # fire the main engine
    obs, r, term, trunc, _ = env.step(a)
    print(t, acts[a], {n: round(float(v), 3) for n, v in zip(names, obs)}, "reward", round(r, 3))
PY
```

Check yourself: which number changes when the main engine fires? Where is the pad (what
are `x` and `y` when you are exactly on it)? Why are the legs 0 here?

The agent never sees pixels, only these 8 numbers. The four actions are on/off: there is
no "half throttle" in the discrete version.

## 3. Interpret episode reward (10 min)

Open `lunarrl/common.py` and `README.md` ("The reward, accurately"), then run:

```bash
python - <<'PY'
import gymnasium as gym, numpy as np
env = gym.make("LunarLander-v3"); rng = np.random.default_rng(0)
obs, _ = env.reset(seed=10000); total = 0; rewards = []
while True:
    obs, r, term, trunc, _ = env.step(int(rng.integers(4)))
    rewards.append(r); total += r
    if term or trunc: break
print("steps", len(rewards), "return", round(total, 1))
print("last reward", rewards[-1], "| sum before the last step", round(total - rewards[-1], 1))
print("biggest single-step rewards", [round(float(r), 1) for r in sorted(rewards)[-3:]])
PY
```

Things to be able to say afterwards:

- The return is the **sum of every step's reward**, not just the ending. A crash episode
  scores about -100 *plus* everything that happened on the way down.
- Most of the signal is **shaping**: each step pays the change in a score for being near
  the pad, slow, level and on its legs. Moving toward the pad earns points immediately.
- The final step is **replaced** by -100 (crash or off screen) or +100 (came to rest).
  "Came to rest" can happen off the pad.
- 200 is Gymnasium's "solved" bar, not a law of physics.

Then open `results/ppo_baseline/training_episodes.csv` in a spreadsheet and plot
`episode_return` against `timesteps_end`. That is the raw data behind the website's
learning curve.

## 4. Change one thing (15 min of mostly waiting)

Pick **one** change. Keep the seed and step count the same as a baseline so the change is
the only difference.

```bash
# baseline at the same short budget
python train.py --run-name learn_base --timesteps 200000 --seed 0

# option A: hyperparameter. No entropy bonus means less exploration.
python train.py --run-name learn_ent0 --timesteps 200000 --seed 0 --ent-coef 0.0

# option B: reward wrapper. Doubles the fuel cost the agent learns from (lunarrl/wrappers.py).
python train.py --run-name learn_fuel --timesteps 200000 --seed 0 --reward-wrapper fuel
```

Before you look at results, write down what you predict and why. For the fuel wrapper:
fewer engine firings, but maybe more hard landings? For `ent-coef 0`: faster at first, or
stuck in a bad habit?

## 5. Compare before and after (10 min)

```bash
python evaluate.py --run-name learn_base --episodes 50 --skip-random
python evaluate.py --run-name learn_fuel --episodes 50 --skip-random    # or learn_ent0
python compare.py learn_base learn_fuel
```

Both evaluations use the same 50 seeds and the **original** reward, even for the wrapped
run, so the numbers are comparable. How to read it honestly:

- Look at outcomes (crashed, landed, timeout), not only the mean.
- A difference of a few points with a standard deviation of 50+ is noise. One training
  seed per setting cannot prove a change is better. To claim it, rerun both with
  `--seed 1` and `--seed 2` and see whether the difference holds every time.
- I ran this fuel experiment at full length (1M steps, one seed). Its result is on the
  project page and in `results/exp_fuel/`. Compare your short run with it, and notice how
  much the step budget matters.

## 6. Three interview questions you should be able to answer honestly

**"What did you actually build, and what did the libraries do?"**
Gymnasium provides the simulator, physics and reward; Stable-Baselines3 provides PPO and
the network. You wrote the training and evaluation pipeline, the same-seed random baseline,
outcome classification from the simulator's flags, the reward-wrapper experiment, the
export, and the charts. The hyperparameters came from rl-baselines3-zoo, not from your own
tuning. Saying this plainly is a strength.

**"How do you know the agent is good and not just lucky?"**
Evaluation is separate from training: 100 episodes on seeds training never used, actions
deterministic, the random policy on the exact same seeds. Report the mean with its standard
deviation and the outcome counts, plus the results of the other training seeds. Then the
limit: a few training seeds is not a confidence interval, and it is all one simulator.

**"Explain the reward. Is it just +100 for landing and -100 for crashing?"**
No. Each step pays the change in a shaped score (distance to pad, speed, tilt, +10 per leg
in contact) minus fuel (0.3 main, 0.03 side per frame). The ±100 replaces the reward on the
final step only. Shaping is what makes it learnable in about a million steps, and it is
also why a high score does not automatically mean a gentle landing on the pad, which is
why outcomes are counted separately.

Bonus, if you have time: why does the training curve sit below the evaluation numbers?
(Training actions are sampled from the policy, so they include exploration; evaluation
picks the most likely action.)
