# Bus: Digital Board Game & RL Environment

A high-performance, deterministic engine and Reinforcement Learning environment for the classic board game **Bus**. 

This project provides a rule-complete implementation of the game, designed from the ground up to support both human play and advanced AI training using self-play and Prioritized Fictitious Self-Play (PFSP).

## 🚀 Overview

- **Deterministic Game Engine**: A Markov-complete state machine that enforces all official rules of *Bus*.
- **Gymnasium Integration**: A custom environment following the Gymnasium API, optimized for Reinforcement Learning.
- **Advanced RL Infrastructure**: Built-in support for:
    - **Action Masking**: Large discrete action space (1,670 actions) handled via `MaskablePPO`.
    - **Self-Play League**: Opponent pool management with Elo rating tracking.
    - **Balanced Matchmaking**: PFSP (Prioritized Fictitious Self-Play) to prevent policy collapse and ensure robust learning.
    - **MCTS Integration**: Monte Carlo Tree Search for both evaluation and hybrid play.
- **Modern GUI**: A visual interface for human play and agent inspection.

## 🛠 Tech Stack

- **Core**: Python 3
- **RL Framework**: Gymnasium, Stable Baselines 3 (SB3 + `sb3-contrib`)
- **Deep Learning**: PyTorch
- **Environment Wrappers**: Custom vectorized and multi-policy wrappers
- **Testing**: Pytest

## 📁 Project Structure

```text
├── core/               # Core game components (Board, Player, GameState)
├── engine/             # Rule enforcement and action resolution logic
├── rl/                 # RL environment, observation/action encoding, and reward logic
├── scripts/            # Training, evaluation, and visualization scripts
├── gui/                # Graphical User Interface
├── data/               # Static board topology and game configuration
├── tests/              # Comprehensive unit and integration test suite
└── bash_scripts/       # Helper scripts for launching training runs
```

## 🏗 Key Components

### Reinforcement Learning (`/rl`)
The RL implementation features a complex observation space (1,458 floats) and a unified discrete action space. It uses **Action Masking** to ensure the agents only consider legal moves in any given phase (Choosing Actions, Resolving Actions, Vrrooomm!, etc.).

### Self-Play System
The project includes a robust self-play training pipeline:
- **Opponent Pool**: Saves historic versions of the agent to provide a diverse training set.
- **Elo Tracker**: Maintains real-time relative strength ratings for all checkpoints.
- **Matchmaking**: Dynamically selects opponents based on win rates (PFSP) to maximize the learning signal.

## 🚦 Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bus-rl.git
   cd bus-rl
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Training an Agent
To start a multi-policy self-play training run with PFSP:
```bash
python scripts/train.py --use-opponent-pool --multi-policy --n-envs 16
```
Or use the provided bash script:
```bash
./bash_scripts/pfsp_ppo.bash
```

### Evaluation
Evaluate a trained model against an MCTS baseline or other checkpoints:
```bash
python scripts/evaluate.py --model-path logs/path_to_model --use-mcts
```

## 📝 Status
The project is currently in an advanced state. Core rules, RL infrastructure, and the self-play training loop are fully implemented and verified with integration tests.

---
*Developed as part of the Bus Digital & RL Project.*
