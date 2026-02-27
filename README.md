# msgym: Gymnasium Environments for MusculoSkeletal Models

[Gymnasium](https://gymnasium.farama.org/) environments for the **MS-Human-700** full-body human musculoskeletal model in [MuJoCo](https://mujoco.readthedocs.io/). 
This repository provides reinforcement learning–ready simulation environments for locomotion and manipulation tasks.

<p align="center">
  <a href="https://lnsgroup.cc/research/MS-Human">Project Page</a> | <a href="https://github.com/LNSGroup/MS-Human-700">MS-Human-700 Model</a>
</p>


<div align="center">
  <img src="pictures/ms_human_render_front.png" width="40%">
  <img src="pictures/render_gif.gif" width="40%">
</div>

## Overview

MS-Human-700 is a whole-body human musculoskeletal model with anatomically detailed body, joint, and muscle parameters. It includes 700 muscle–tendon units.

Related papers:
- [MS-Human-700 (ICRA 2024)](https://arxiv.org/abs/2312.05473)
- [DynSyn (ICML 2024)](https://arxiv.org/abs/2407.11472)
- [MPC2 (ICLR 2025)](https://arxiv.org/abs/2505.08238)
- [QFlex (ICLR 2026)](https://arxiv.org/abs/2601.19707)
- [and more (on balance & fall, contact-rich & deformable, vision language model ...)](https://lnsgroup.cc/research/MS-Human#papers)

This repository wraps MS-Human-700 as a Python package **msgym** so you can use it with `gymnasium.make()` after a standard install.

## Installation

### 1. Clone the repository with submodules

The MuJoCo model and assets live in the **MS-Human-700** submodule. Clone with recursion so it is included:

```bash
git clone https://github.com/LNSGroup/msgym.git --recursive 
cd msgym
```

If you already cloned without `--recursive`, initialize and update submodules:

```bash
git submodule update --init --recursive
```

### 2. Install the package

We recommend using [uv](https://github.com/astral-sh/uv) for fast dependency management (Python 3.12 by default):

```bash
uv python pin 3.12
uv sync
uv pip install -e .
```

After this, `import msgym` and `gymnasium.make("msgym/...")` work in your environment.

### Optional: DynSyn-SAC training

For [DynSyn](https://github.com/Beanpow/DynSyn) reinforcement learning training, install the optional dependency set.

```bash
uv sync --extra dynsyn
```

## Usage

Environments are registered with Gymnasium on `import msgym`. Use them as follows:

```python
import gymnasium as gym
import msgym

env = gym.make("msgym/LocomotionFullEnv-v1", render_mode="human", gait_cycles=5)
obs, info = env.reset()
# ... step, render
env.close()
```

**Registered environment IDs:**

| Environment ID                    | Description                          |
|-----------------------------------|--------------------------------------|
| `msgym/LocomotionFullEnv-v1`      | Full-body locomotion imitation       |
| `msgym/LocomotionLegsEnv-v1`      | Legs-only locomotion imitation       |
| `msgym/ManipulationEnv-v1`         | Right-arm manipulation (grasp/lift)   |

Run the test script from the repo root:

```bash
uv run python env_test.py
```

> **Headless / no display:**  
> 1. Use `render_mode="rgb_array"` (or omit rendering).  
> 2. For offscreen rendering, set `MUJOCO_GL=egl` (e.g. `export MUJOCO_GL=egl` or `$env:MUJOCO_GL="egl"` on Windows).

## Training (DynSyn-SAC)

Controllers can be trained with the DynSyn-SAC algorithm.

**Training:**

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run python DynSyn-SAC/SB3-Scripts/train.py -f DynSyn-SAC/configs/locomotionFull.json
```

**Evaluation:**

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run python DynSyn-SAC/SB3-Scripts/eval.py -f DynSyn-SAC/logs/LocomotionFull
```

Trained checkpoints are available from GitHub Releases.

## License

This project is released under the [Apache-2.0 License](LICENSE).

## Citation

If you use MS-Human-700 or msgym in academic work, please cite:

```bibtex
@inproceedings{zuo2024self,
  title={Self model for embodied intelligence: Modeling full-body human musculoskeletal system and locomotion control with hierarchical low-dimensional representation},
  author={Zuo, Chenhui and He, Kaibo and Shao, Jing and Sui, Yanan},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={13062--13069},
  year={2024},
  organization={IEEE}
}
```

## Control Demo

[**DynSyn**](https://github.com/Beanpow/DynSyn) control results:

<div align="center">
  <img src="pictures/loco_full_gif.gif" width="32%">
  <img src="pictures/loco_legs_gif.gif" width="32%">
  <img src="pictures/mani_gif.gif" width="32%">
</div>

[**QFlex**](https://lnsgroup.cc/research/Qflex) control results:

<div align="center">
  <img src="pictures/run_gif.gif" width="49%">
  <img src="pictures/dance_gif.gif" width="49%">
</div>

**High-Fidelity Motion Tracking** results: 

Leveraging MuJoCo Warp for massively parallel GPU simulation enables the rapid and efficient training of control policies capable of high-precision motion tracking across diverse and dynamic trajectories.

The demos below illustrate these tracking capabilities of the MS-Human model:
*   **Overlap**: The model and reference trajectory are rendered directly to visualize tracking accuracy.
*   **Separate**: The model and reference trajectory are rendered with an offset to showcase motion details.

<table>
  <tr>
    <td align="center" width="25%">
      <img src="https://github.com/user-attachments/assets/0350ff08-2ab3-4019-a4be-bef6fbc276ad" alt="running_overlap" width="100%"><br>
      Running: Overlap
    </td>
    <td align="center" width="25%">
      <img src="https://github.com/user-attachments/assets/16084e8c-41f5-45a3-96b7-b69307c78e0c" alt="running_separate" width="100%"><br>
      Running: Separate
    </td>
  </tr>
  <tr>
    <td align="center" width="25%">
      <img src="https://github.com/user-attachments/assets/9aea44e9-3277-4d81-a096-6865e2dcc5e4" alt="walking_ovelap" width="100%"><br>
      Walking: Overlap
    </td>
    <td align="center" width="25%">
      <img src="https://github.com/user-attachments/assets/9c32033c-0ca6-4b0c-a6fd-f61f66481ee9" alt="walking_separate" width="100%"><br>
      Walking: Separate
    </td>
  </tr>
</table>

[**Contact-Rich Deformable Interaction**](https://lnsgroup.cc/research/SoftHand) simulation results:

<table>
  <tr>
    <td align="center" width="40%">
      <img src="https://github.com/user-attachments/assets/e152a6ad-91c3-469f-ae5f-3a4f4747ac82" alt="softfoot" width="100%">
    </td>
    <td align="center" width="55%">
      <img src="https://github.com/user-attachments/assets/ccc97dc9-d05b-4982-b765-a3e761a6d604" alt="softhand" width="100%">
    </td>
  </tr>
</table>
