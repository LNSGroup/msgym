from typing import Any, Callable, List, Optional
import os
import sys
import site
import numpy as np
from gymnasium import spaces
import mujoco

def action_obs_check(cls: Any) -> None:
    """Verify that action and observation spaces have distinct low/high bounds.

    Args:
        cls: Environment class with action_space and observation_space attributes.

    Raises:
        ValueError: If any dimension of action or observation space has low == high.
    """
    low = cls.action_space.low
    high = cls.action_space.high
    if (low == high).any():
        raise ValueError("Action space has the same low and high value")

    low = cls.observation_space.low
    high = cls.observation_space.high
    if (low == high).any():
        raise ValueError("Observation space has the same low and high value")


def get_ms_human_model_path(filename: str) -> str:
    """Resolve path to an MS-Human-700 XML model file.

    Tries multiple locations:
    1. Relative to the source tree (for editable / local installs).
    2. Under sys.prefix/MS-Human-700 (for some data_files installs).
    3. Under each site-packages/dist-packages directory.
    """
    tried: List[str] = []

    # 1. Source / editable install: project_root/MS-Human-700/<filename>
    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidate = os.path.join(src_root, "MS-Human-700", filename)
    tried.append(candidate)
    if os.path.exists(candidate):
        return candidate

    # 2. Installed via data_files: sys.prefix/MS-Human-700/<filename>
    candidate = os.path.join(sys.prefix, "MS-Human-700", filename)
    tried.append(candidate)
    if os.path.exists(candidate):
        return candidate

    # 3. site-packages / dist-packages locations (e.g. Colab)
    for base in site.getsitepackages() + [site.getusersitepackages()]:
        if not base:
            continue
        candidate = os.path.join(base, "MS-Human-700", filename)
        tried.append(candidate)
        if os.path.exists(candidate):
            return candidate

    tried_str = "\n".join(f"- {path}" for path in tried)
    raise ValueError(f"Could not locate MS-Human-700 model file. Tried:\n{tried_str}")

def get_observation_space(
    xml_path: str,
    get_obs_fn: Callable[..., np.ndarray],
    obs_kwargs: Optional[dict] = None,
) -> spaces.Box:
    """Build a Box observation space from an XML model and observation function.

    Args:
        xml_path: Path to MuJoCo XML model file.
        get_obs_fn: Function that takes mujoco.MjData and optional kwargs, returns 1D obs.
        obs_kwargs: Optional keyword arguments passed to get_obs_fn.

    Returns:
        Gymnasium Box observation space with shape inferred from get_obs_fn output.
    """
    if mujoco is None:
        raise ImportError("MuJoCo is required for get_observation_space.")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    obs = get_obs_fn(data, **(obs_kwargs or {}))
    assert obs.ndim == 1, "Observation must be 1D"
    return spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs.shape[0],), dtype=np.float64
    )

def get_render_fps(xml_path: str, skip_frames: int) -> int:
    """Compute effective render FPS from model timestep and frame skip.

    Args:
        xml_path: Path to MuJoCo XML model file.
        skip_frames: Number of simulation steps per environment step.

    Returns:
        Rounded FPS (1 / (timestep * skip_frames)).
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    timestep = model.opt.timestep
    return int(round(1.0 / timestep / skip_frames))

def euler2quat(euler: np.ndarray) -> np.ndarray:
    """Convert Euler angles (ZYX order) to quaternions (w, x, y, z).

    Args:
        euler: Array of shape (..., 3) with Euler angles in radians.

    Returns:
        Array of shape (..., 4) with quaternions (w, x, y, z).
    """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, f"Invalid shape euler {euler.shape}"

    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quat

def joint_name_to_dof_index(
    all_joint_name_list: List[str],
    joint_name_list: List[str],
) -> List[int]:
    """Map joint names to indices in a full joint name list.

    Args:
        all_joint_name_list: Full list of joint names (order defines indices).
        joint_name_list: Subset of joint names to look up.

    Returns:
        List of indices for each name in joint_name_list.

    Raises:
        ValueError: If any name in joint_name_list is not in all_joint_name_list.
    """
    joint_index_list = []
    for joint_name in joint_name_list:
        if joint_name in all_joint_name_list:
            joint_index_list.append(all_joint_name_list.index(joint_name))
        else:
            raise ValueError(
                f"Joint name {joint_name} not found in all joint name list"
            )
    return joint_index_list
