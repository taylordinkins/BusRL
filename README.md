# Bus: Digital Board Game & RL Environment

A high-performance, deterministic engine and reinforcement-learning environment for the board game **Bus**.

Built almost exclusively with AI assistance (Claude, GPT, Gemini).

The project provides a rule-complete implementation of the game designed to support both human play and agent training via multi-policy self-play with Prioritized Fictitious Self-Play (PFSP).

## 🚀 Overview

- **Deterministic Game Engine** — a Markov-complete state machine that enforces the official rules of *Bus*.
- **Gymnasium Environment** — a custom `BusEnv` following the Gymnasium API, optimized for RL.
- **Hierarchical, Phase-Aware Policy** — one shared network split into ten decision heads, each with its own compact, legally-masked action catalog (see below).
- **Self-Play League** — an opponent pool of historic checkpoints with PFSP matchmaking and OpenSkill/Elo skill tracking.
- **Desktop GUI** — a PySide6 interface for human play and agent inspection.

## 🛠 Environment & Tech Stack

- **Language**: Python 3.14 (Python 3.10+ generally works)
- **RL Framework**: Gymnasium, Stable-Baselines 3 (`stable-baselines3` + `sb3-contrib` `MaskablePPO`)
- **Deep Learning**: PyTorch (CPU or CUDA; device auto-detected)
- **Skill Tracking**: OpenSkill (Plackett–Luce) or Elo
- **GUI**: PySide6
- **Graph / Plotting**: NetworkX, Matplotlib
- **Testing**: Pytest

Training runs write TensorBoard logs, checkpoints, best models, and the opponent pool under `logs/<run_name>/`. Compute is device-agnostic — `--device auto` selects CUDA when available and falls back to CPU. All engine state transitions are deterministic, so runs are reproducible given a seed.

## 📁 Project Structure

```text
├── core/          # Core game components (Board, Player, GameState)
├── engine/        # Rule enforcement and action resolution
├── rl/            # Environment, observation/action encoding, policy, self-play infra
├── scripts/       # Training and evaluation entrypoints
├── gui/           # PySide6 graphical interface
├── data/          # Static board topology and game configuration
├── tests/         # Unit and integration test suite
├── bash_scripts/  # Launchers for full training runs
└── logs/          # Per-run TensorBoard logs, checkpoints, and opponent pools
```

## 🏗 Reinforcement Learning (`/rl`)

### Hierarchical, phase-aware action space

Rather than a single flat action space, decisions are routed to one of **ten heads** based on the current game context (`rl/hierarchical_action_space.py`):

| Head | Context |
|------|---------|
| `SETUP_BUILDINGS` | Initial building placement |
| `SETUP_RAILS_FORWARD` / `SETUP_RAILS_REVERSE` | Initial rail placement passes |
| `CHOOSING_ACTIONS` | Placing action markers / passing |
| `RESOLVE_LINE_EXPANSION` | Resolving line expansion |
| `RESOLVE_PASSENGERS` | Resolving passenger distribution |
| `RESOLVE_BUILDINGS` | Resolving building placement |
| `RESOLVE_TIME_CLOCK` | Advancing or stopping the clock |
| `RESOLVE_VRROOMM_PASSENGER` / `RESOLVE_VRROOMM_DEST` | The two Vrroomm! sub-decisions |

The active head is selected deterministically from the phase, resolution area, and Vrroomm! stage via `get_head_id`. Each head owns a fixed action catalog; the environment exposes a single `Discrete(216)` action space (the largest catalog), and `ActionMaskGenerator` masks it down to the current head's legal actions every step. The flat **1,470-float** observation includes a one-hot head id and Vrroomm! stage, conditioning the shared trunk on which decision it is currently making.

A single `MaskablePPO` policy (`BusMaskableActorCriticPolicy`) learns all heads jointly, with logit clamping and finite-value sanity checks for training stability.

Two optional observation feature groups can be enabled at training time: `--use-slot-actionability` and `--use-delivery-features`.

### Self-play system

- **Opponent Pool** (`rl/opponent_pool.py`) — snapshots historic checkpoints for a diverse training set, with configurable pruning.
- **Skill Tracking** (`rl/openskill_tracker.py`, `rl/elo_tracker.py`) — maintains relative strength ratings across checkpoints.
- **PFSP Matchmaking** (`rl/multi_policy_env.py`) — samples opponents by strength to maximize learning signal and prevent policy collapse.
- **Reward shaping** (`rl/reward.py`) — configurable per-step and terminal rewards, including a score-based terminal mode that rewards absolute delivery count rather than margin of victory.

## 🚦 Getting Started

### Installation

```bash
git clone https://github.com/yourusername/bus-rl.git
cd bus-rl
pip install -r requirements.txt
```

### Training an agent

Training runs are driven by `scripts/train.py`. The canonical, documented setup lives in `bash_scripts/pfsp_ppo_v2.bash` — start there. Its Phase 1 primes a fresh policy under a sparse, delivery-centric reward regime with 100% self-play:

```bash
python scripts/train.py \
    --use_opponent_pool \
    --multi_policy \
    --self_play_prob 1.0 \
    --sampling_method pfsp \
    --pool_size 120 \
    --total_timesteps 8000000 \
    --n_envs 16 \
    --n_steps 4096 \
    --batch_size 2048 \
    --lr 5e-4 \
    --gamma 0.995 \
    --gae_lambda 1.0 \
    --vf_coef 1.0 \
    --ent_coef 0.02 --ent_coef_final 0.006 \
    --target_kl 0.015 \
    --skill_tracking openskill \
    --use_score_based_terminal --terminal_score_scale 1.0 \
    --delivery_reward 2.5 \
    --station_connection_reward 0.05 \
    --use-slot-actionability \
    --use-delivery-features
```

Or just run the script:

```bash
./bash_scripts/pfsp_ppo_v2.bash
```

Phase 2 (commented out at the bottom of the same script) resumes from Phase 1's best model and pool, drops self-play to 20% to introduce adversarial PFSP pressure, and enables periodic pool evaluation for OpenSkill updates. Edit `PHASE1_RUN` to point at the completed Phase 1 log directory before running it.

**What to watch (TensorBoard):** `eval/mean_game_score` and `rollout/mean_game_score` (ground-truth points), `train/value_loss` (should decline steadily), `train/approx_kl` (should stay active), and `rollout/ep_len_true` (~110–140 steps is healthy).

### Evaluation

Play a trained checkpoint and report scores:

```bash
python scripts/evaluate.py logs/<run_name>/best_model/best_model.zip --num-games 20
```

## 📝 Status

Core rules, the hierarchical RL environment, and the multi-policy self-play loop are implemented and covered by integration tests. Active development focuses on reward shaping and PFSP dynamics under the score-based terminal regime.

## 🔭 To-Do

- **Port to OpenSpiel** — expose the engine as an [OpenSpiel](https://github.com/google-deepmind/open_spiel) game to reuse its algorithm suite (CFR, PSRO, MCTS, etc.) and standardized self-play tooling.
