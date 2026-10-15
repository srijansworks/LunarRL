# LunarRL: Teaching an Agent to Land

PPO (Stable-Baselines3) trained on Gymnasium's `LunarLander-v3`, evaluated against
a uniform-random policy on the same held-out seeds. This folder is self-contained;
the portfolio page `project.html?id=lunar-rl` draws its charts from files this code
generates.

```
lunar-rl/
├─ train.py            train PPO, save model + per-episode training log
├─ evaluate.py         PPO vs random on the same N seeds → eval_summary.json / eval_episodes.csv
├─ export_results.py   same-seed replay MP4s + the data file the website reads
├─ compare.py          print several runs side by side (for experiments)
├─ watch.py            watch one episode in a window
├─ tests/              fast checks: python -m unittest discover -s tests
├─ lunarrl/            shared helpers: env factory, outcome classification, reward wrappers
├─ results/            outputs of the published runs (tracked in git)
├─ LEARN-LUNARRL.md    one-hour guided walkthrough
├─ models/             saved weights (ignored by git; included in the zip for convenience)
└─ runs/               raw Monitor logs (ignored by git)
```

## Published results

Measured, not estimated: copied from `results/*/eval_summary.json`. 100 episodes per row,
evaluation seeds 10000..10099, PPO trained for 1,015,808 steps per run.

| policy | mean ± SD | solved (≥200) | landed on pad | crashed | off screen | timeout | main-engine frames |
|---|---|---|---|---|---|---|---|
| random policy | -180.5 ± 105.5 | 0 | 0 | 90 | 10 | 0 | 23.3 |
| PPO seed 0 (published) | 256.8 ± 25.9 | 99 | 98 | 0 | 0 | 1 | 165.2 |
| PPO seed 1 | 261.6 ± 20.9 | 100 | 97 | 0 | 0 | 0 | 161.1 |
| PPO seed 2 | 266.6 ± 21.2 | 100 | 96 | 0 | 0 | 0 | 141.1 |
| PPO seed 0 + fuel wrapper | 258.6 ± 45.5 | 95 | 84 | 2 | 0 | 3 | 123.1 |

The fuel-wrapper run used about a quarter fewer main-engine frames but landed between
the flags less often. With one seed per setting that is a lead, not a conclusion.

## Setup (macOS, Apple Silicon or Intel)

Needs Python 3.11. All dependencies ship prebuilt macOS wheels, so no `swig` or
compiler is needed.

```bash
brew install python@3.11            # skip if `python3.11 --version` already works
cd lunar-rl
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import gymnasium as g; e=g.make('LunarLander-v3'); print(e.observation_space, e.action_space)"
```

The last line should print an 8-dimensional `Box` and `Discrete(4)`. Then
`python -m unittest discover -s tests` should report 5 tests OK in a few seconds.

## Reproduce the published run

```bash
python train.py --run-name ppo_baseline --seed 0          # 1M steps; took 6 min here (4-core Linux CPU)
python evaluate.py --run-name ppo_baseline --episodes 100 # PPO and random, seeds 10000..10099
python train.py --run-name ppo_seed1 --seed 1             # optional: more training seeds
python train.py --run-name ppo_seed2 --seed 2
python train.py --run-name exp_fuel --seed 0 --reward-wrapper fuel   # the experiment
python evaluate.py --run-name ppo_seed1 --skip-random
python evaluate.py --run-name ppo_seed2 --skip-random
python evaluate.py --run-name exp_fuel --skip-random
python export_results.py --extra-runs ppo_seed1 ppo_seed2 --experiment exp_fuel
```

`export_results.py` writes `results/<run>/site_data.json` and replays, then copies
`assets/data/lunarrl-results.js` and `assets/media/lunar-rl-{ppo,random}.{mp4,jpg}`
into the portfolio (the folder above `lunar-rl/`). Refresh the page to see your run.
Drop `--experiment exp_fuel` if you haven't trained that run, and `--extra-runs`
if you only trained one seed.

Quick check that everything works (about a minute, the agent will still be bad):

```bash
python train.py --run-name quick --timesteps 50000
python evaluate.py --run-name quick --episodes 10
```

To watch the agent fly in a window: `python watch.py` (or `python watch.py --random`).

## The reward, accurately

The reward is Gymnasium's, unchanged (see `gymnasium/envs/box2d/lunar_lander.py`).
Each step pays the **change** in a shaping score

    shaping = -100·distance_to_pad - 100·speed - 100·|angle| + 10·left_leg + 10·right_leg

minus fuel: 0.3 per frame of main engine, 0.03 per frame of a side engine. On the
final step the reward is **set to** -100 if the body touches the ground or the lander
leaves the screen (`|x| ≥ 1`), or +100 if the body comes to rest (Box2D puts it to
sleep). "Coming to rest" does not require being on the pad; the distance term is what
pulls it there. Episodes are cut at 1,000 steps. A return ≥ 200 counts as solved.

## How outcomes are counted

`lunarrl/common.py: classify()` reads the simulator's own flags after the final step:
`timeout` (truncated at 1,000 steps), `crashed` (`game_over`), `out_of_bounds`
(`|x| ≥ 1`), otherwise `landed`. A landing is `on_pad` when both legs touch and
`|x| ≤ 0.2` (the flags).

## Seeds and determinism

- Training: `--seed` seeds Python, NumPy, PyTorch, and the 16 environments (env *i* gets `seed + i`).
- Evaluation: seeds 10000..10099 by default, never used in training. PPO acts deterministically
  (most likely action). The random policy uses a NumPy generator seeded with the first eval seed.
- Re-running on the same machine with the same package versions reproduces the numbers. Checked here:
  two 49k-step trainings with the same seed wrote byte-identical training logs, and re-running
  `evaluate.py` reproduced every evaluation number. On a
  different CPU or OS, PyTorch floating-point differences can change training, so expect similar,
  not identical, results.

## Experiment: change one thing, compare

```bash
python train.py --run-name exp_fuel --reward-wrapper fuel    # doubles the fuel cost the agent learns from
python evaluate.py --run-name exp_fuel --skip-random
python compare.py ppo_baseline exp_fuel
```

Other one-flag experiments: `--ent-coef 0.0`, `--gamma 0.99`, `--learning-rate 1e-4`,
`--timesteps 300000`, or `--reward-wrapper softcrash`. Evaluation always uses the original
reward, so the comparison is fair. Change one thing at a time and keep the seed.

## Licences and attribution

- Gymnasium (MIT), Farama Foundation. LunarLander environment and reward function.
- Box2D (zlib licence) physics, via the `box2d` Python bindings.
- Stable-Baselines3 (MIT), DLR-RM. PPO implementation.
- PyTorch (BSD-style licence).
- Hyperparameters from rl-baselines3-zoo (MIT), `hyperparams/ppo.yml`, LunarLander entry.
- Replay frames are rendered by Gymnasium's own renderer.
- PPO: Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
