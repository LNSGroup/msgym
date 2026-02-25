from typing import Any, Dict, Optional
import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecVideoRecorder
from wrapper import *


def _ensure_env_registered(env_name: str) -> None:
    """Ensure Gymnasium env is registered.

    For msgym environments, registration happens on `import msgym`.
    This function is safe to call in subprocesses (SubprocVecEnv).
    """
    if isinstance(env_name, str) and env_name.startswith("msgym/"):
        try:
            import msgym  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Environment id starts with 'msgym/', but msgym is not installed. "
                "Install it (e.g. `pip install -e .` at repo root) before training/eval."
            ) from exc

def create_env(
    env_name: str,
    single_env_kwargs: Dict[str, Any],
    wrapper_list: Dict[str, Any],
    seed: int = 0,
    render_mode: Optional[str] = None,
) -> gym.Env:
    """Create a single environment with optional wrappers.

    Args:
        env_name: Gymnasium environment ID.
        single_env_kwargs: Keyword arguments passed to gym.make.
        wrapper_list: Dict mapping wrapper class names to their kwargs.
        seed: Random seed for the environment.
        render_mode: Render mode passed to gym.make.

    Returns:
        The wrapped environment.
    """
    _ensure_env_registered(env_name)
    set_random_seed(seed)
    env = gym.make(env_name, render_mode=render_mode, **single_env_kwargs)
    for wrapper_name, wrapper_args in wrapper_list.items():
        try:
            env = eval(wrapper_name)(env, **wrapper_args)
        except NameError:
            print(f"Wrapper {wrapper_name} not found!")
            raise
    return env

def create_vec_env(
    env_name: str,
    single_env_kwargs: Dict[str, Any],
    env_nums: int,
    wrapper_list: Optional[Dict[str, Any]] = None,
    monitor_dir: Optional[str] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    render_mode: Optional[str] = None,
) -> VecEnv:
    """Create a vectorized environment with optional monitoring."""
    if wrapper_list is None:
        wrapper_list = {}
    if monitor_kwargs is not None and hasattr(monitor_kwargs, "info_keywords"):
        monitor_kwargs["info_keywords"] = tuple(monitor_kwargs["info_keywords"])
    vec_env = make_vec_env(
        create_env,
        env_kwargs={
            "env_name": env_name,
            "single_env_kwargs": single_env_kwargs,
            "wrapper_list": wrapper_list,
            "seed": seed,
            "render_mode": render_mode,
        },
        n_envs=env_nums,
        vec_env_cls=SubprocVecEnv,
        monitor_dir=monitor_dir,
        monitor_kwargs=monitor_kwargs,
    )
    return vec_env

def record_video(
    vec_norm: Any,
    model: Any,
    args: Any,
    video_dir: str,
    video_ep_num: int,
    name_prefix: str = "video",
) -> None:
    """Record evaluation videos of the agent in the environment.

    Args:
        vec_norm: VecNormalize wrapper (or vec env) used for observation normalization.
        model: Trained SB3 model.
        args: Namespace with env_name, single_env_kwargs, wrapper_list, seed.
        video_dir: Directory to save video files.
        video_ep_num: Number of episodes to record.
        name_prefix: Prefix for video filenames.
    """
    env = create_vec_env(
        args.env_name,
        args.single_env_kwargs,
        1,
        wrapper_list=args.wrapper_list,
        monitor_dir=None,
        monitor_kwargs=None,
        seed=args.seed,
        render_mode="rgb_array",
    )
    vec_env = VecVideoRecorder(
        env,
        video_folder=video_dir,
        record_video_trigger=lambda x: True,
        video_length=10000,
        name_prefix=name_prefix,
    )
    for _ in range(video_ep_num):
        total_reward: float = 0.0
        obs = vec_env.reset()
        done = False
        while not done:
            obs = vec_norm.normalize_obs(obs)
            action, _ = model.predict(obs, deterministic=False)
            obs, r, done, info = vec_env.step(action)
            total_reward += float(np.asarray(r).sum())
        print(f"Episode reward: {total_reward}")
        vec_env._stop_recording()
    vec_env.close()
