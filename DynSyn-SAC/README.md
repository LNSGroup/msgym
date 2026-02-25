# DynSyn-SAC: Dynamical Synergistic Representation for Efficient Learning and Control

SAC-based training implementation of the **DynSyn** algorithm for overactuated embodied systems. For synergy generation and other backbones, see [DynSyn](https://github.com/Beanpow/DynSyn).

**Homepage:** [sites.google.com/view/dynsyn](https://sites.google.com/view/dynsyn) · **Publication:** [DynSyn (ICML 2024)](https://dl.acm.org/doi/10.5555/3692070.3692797)

---

## Installation

### Option A: From repo root (recommended)

If you're using DynSyn-SAC with the `msgym` environments in this repository, install everything from the project root:

```bash
uv sync --extra dynsyn
uv pip install -e .
```

### Option B: Install DynSyn-SAC only (standalone venv)

DynSyn-SAC is also pip-installable as a standalone package (installs the `DynSyn` module and RL deps):

```bash
uv venv
uv pip install -e DynSyn-SAC
```

Note: to train on `msgym/...` environments, you still need `msgym` installed (Option A, or `uv pip install -e .` at repo root).

---

## Training

Train an agent with a config file:

```bash
python SB3-Scripts/train.py -f PATH_TO_CONFIG_FILE
```

From the repo root with `uv`:

```bash
uv run python DynSyn-SAC/SB3-Scripts/train.py -f DynSyn-SAC/configs/locomotionFull.json
```

Example (with msgym environments and models set up):

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run python DynSyn-SAC/SB3-Scripts/train.py -f DynSyn-SAC/configs/locomotionFull.json
```

On headless servers or without a display, set `MUJOCO_GL=egl` for offscreen rendering.

---

## Evaluation

Evaluate a trained model and record videos:

```bash
python SB3-Scripts/eval.py -f PATH_TO_LOG_DIR [-m PATH_TO_MODEL] [-n NUM_EPISODES]
```

From the repo root with `uv`:

```bash
uv run python DynSyn-SAC/SB3-Scripts/eval.py -f DynSyn-SAC/logs/LocomotionFull -n 3
```

- `-f` / `--log_path`: Path to the log directory containing the saved config and checkpoints.
- `-m` / `--model_path`: (Optional) Path to a specific model file; otherwise uses `best_model.zip` in the log dir.
- `-n` / `--num_episodes`: Number of episodes to record.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run python DynSyn-SAC/SB3-Scripts/eval.py -f DynSyn-SAC/logs/LocomotionFull -n 3
```

---

## Citation

If you use DynSyn in your work, please cite:

```bibtex
@inproceedings{he2024dynsyn,
  title={DynSyn: Dynamical Synergistic Representation for Efficient Learning and Control in Overactuated Embodied Systems},
  author={He, Kaibo and Zuo, Chenhui and Ma, Chengtian and Sui, Yanan},
  booktitle={International Conference on Machine Learning},
  pages={18115--18132},
  year={2024},
  organization={PMLR}
}
```
