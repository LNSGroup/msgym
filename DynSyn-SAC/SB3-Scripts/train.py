"""
Training script for reinforcement learning agents using Stable-Baselines3.
This script handles the training process for various RL algorithms, including
custom agents, with support for continuing training from checkpoints.

Usage:
    python train.py --config_file PATH_TO_CONFIG_FILE
"""

import argparse
import importlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, List
import gymnasium
import sb3_contrib
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
from DynSyn import SAC_DynSyn
from callback import (
    SaveConfigToTensorboardCallback,
    SaveVecNormalizeOnBestCallback,
    TensorboardCallback,
    VideoRecorderCallback,
)
from schedule import linear_schedule
from utils import create_vec_env

_CUSTOM_AGENTS = {
    "SAC_DynSyn": SAC_DynSyn,
}


def _ensure_env_registered(env_name: str) -> None:
    if isinstance(env_name, str) and env_name.startswith("msgym/"):
        import msgym  # noqa: F401

def load_policy(args: argparse.Namespace) -> Any:
    policy = args.agent_kwargs.pop("policy", None)
    policy = "MlpPolicy" if policy is None else policy
    if policy != "MlpPolicy":
        policy = eval(policy)
    return policy

def register_callback(
    args: argparse.Namespace,
    video_dir: str,
    log_dir: str,
    config_str: str,
    eval_env: Any,
    checkpoint_dir: str,
) -> List[Any]:
    # Callback
    callback_list = []
    # Convert to total steps
    args.check_freq //= args.env_nums
    args.record_freq //= args.env_nums
    args.dump_freq //= args.env_nums
    callback_list.append(SaveConfigToTensorboardCallback(log_dir, config_str))

    if args.check_freq > 0:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=checkpoint_dir,
            log_path=os.path.join(log_dir, "eval"),
            eval_freq=args.check_freq,
            n_eval_episodes=getattr(args, "eval_episodes", 3),
            deterministic=True,
            render=False,
            callback_on_new_best=SaveVecNormalizeOnBestCallback(
                save_path=os.path.join(checkpoint_dir, "best_env.zip"),
                verbose=1,
            ),
            verbose=1,
        )
        callback_list.append(eval_callback)

    if args.dump_freq > 0:
        callback_list.append(
            CheckpointCallback(
                save_freq=args.dump_freq,
                save_path=checkpoint_dir,
                name_prefix="model",
                save_replay_buffer=getattr(args, "save_replay_buffer", False),
                verbose=1,
            )
        )
    if args.record_freq > 0:
        video_callback = VideoRecorderCallback(args, args.record_freq, video_dir=video_dir, video_ep_num=5, verbose=1)
        callback_list.append(video_callback)

    callback_list.append(TensorboardCallback(getattr(args, "info_keywords", {}), reward_freq=args.reward_freq))

    return callback_list

def build_env(args: argparse.Namespace, monitor_dir: str) -> Any:
    # Parallel environments
    vec_env = create_vec_env(
        args.env_name,
        args.single_env_kwargs,
        args.env_nums,
        wrapper_list=args.wrapper_list,
        monitor_dir=monitor_dir,
        monitor_kwargs=getattr(args, "monitor_kwargs", {}),
        seed=args.seed
    )
    # Vec Norm
    if args.vec_normalize["is_norm"] and not args.load_model_dir:
        vec_env = VecNormalize(vec_env, **args.vec_normalize["kwargs"])
    return vec_env


def build_eval_env(args: argparse.Namespace) -> Any:
    """Build a separate eval env for EvalCallback.

    Uses 1 env, no rendering, and mirrors VecNormalize settings (training=False).
    """
    eval_env = create_vec_env(
        args.env_name,
        args.single_env_kwargs,
        1,
        wrapper_list=args.wrapper_list,
        monitor_dir=None,
        monitor_kwargs=None,
        seed=args.seed + 1000,
        render_mode=None,
    )
    if args.vec_normalize["is_norm"]:
        eval_env = VecNormalize(eval_env, training=False, **args.vec_normalize["kwargs"])
    return eval_env


def find_env_file(env_name: str) -> Any:
    """Locate the source file defining the given Gymnasium environment."""
    _ensure_env_registered(env_name)
    env_spec = gymnasium.spec(env_name)
    module_path, class_name = env_spec.entry_point.split(":")

    module = importlib.import_module(module_path)
    env_dir = os.path.dirname(module.__file__)

    init_file = os.path.join(env_dir, "__init__.py")
    if os.path.exists(init_file):
        with open(init_file, 'r') as f:
            init_content = f.read()

        import_lines = [line.strip() for line in init_content.split('\n') if class_name in line]
        if import_lines:
            for line in import_lines:
                if 'from' in line and 'import' in line:
                    from_part = line.split('from')[1].split('import')[0].strip()
                    module_name = from_part.split('.')[-1]
                    env_file = os.path.join(env_dir, f"{module_name}.py")
                    if os.path.exists(env_file):
                        return env_file
                    else:
                        raise NotImplementedError
            
    return None

def train(args: argparse.Namespace, config_str: str) -> None:
    """Run training: set up dirs, build env, create/load agent, train, and save."""
    # Set up new training session
    log_name = args.config_name
    env_name_log = args.env_name.split("/")[-1]
    
    # Define log directories
    log_dir = os.path.join(args.log_root_dir, env_name_log, time.strftime("%m%d-%H%M%S") + '_' + str(args.seed))
    
    # Create necessary directories
    monitor_dir = os.path.join(log_dir, "monitor")
    checkpoint_dir = os.path.join(log_dir, "checkpoint")
    video_dir = os.path.join(log_dir, "video")
    for directory in [log_dir, monitor_dir, checkpoint_dir, video_dir]:
        os.makedirs(directory, exist_ok=True)
        
    with open(os.path.join(log_dir, log_name + ".json"), "w") as f:
        f.write(config_str)

    # Save environment file (best-effort)
    env_file_path = find_env_file(args.env_name)
    if env_file_path:
        env_file_name = os.path.basename(env_file_path)
        target_path = os.path.join(log_dir, env_file_name)
        shutil.copy2(env_file_path, target_path)
        print(f"Environment file copied to: {target_path}")
    else:
        print(f"Warning: Could not find environment file for {args.env_name}")
        
    # Build eval env for proper best-model selection (no monitor file reads)
    eval_env = build_eval_env(args)

    # Setup callbacks
    callback_list = register_callback(args, video_dir, log_dir, config_str, eval_env=eval_env, checkpoint_dir=checkpoint_dir)

    # Initialize agent class
    if hasattr(sb3, args.agent) or hasattr(sb3_contrib, args.agent):
        Agent = getattr(sb3_contrib, args.agent, getattr(sb3, args.agent, None))
    else:
        Agent = _CUSTOM_AGENTS.get(args.agent)
        if Agent is None:
            Agent = eval(args.agent)

    # Configure learning parameters
    if "learning_rate" in args.agent_kwargs and not isinstance(args.agent_kwargs["learning_rate"], float):
        args.agent_kwargs["learning_rate"] = eval(args.agent_kwargs["learning_rate"])
    args.agent_kwargs["seed"] = args.seed

    # Build environment
    vec_env = build_env(args, monitor_dir)

    # Load or create model
    if args.load_model_dir:
        if os.path.isdir(args.load_model_dir):
            env_path = os.path.join(args.load_model_dir, 'best_env.zip')
            model_path = os.path.join(args.load_model_dir, "best_model.zip")
        else:
            env_path = args.load_model_dir.replace("model", "env")
            model_path = args.load_model_dir
        print(f"Loading model from {model_path}")
        vec_env = VecNormalize.load(env_path, vec_env)
        model = Agent.load(model_path, env=vec_env, verbose=1, tensorboard_log=log_dir, **args.load_kwargs if hasattr(args, "load_kwargs") else {})
                         
        # Update model parameters
        model.learning_rate = args.agent_kwargs["learning_rate"]
        model._setup_lr_schedule()
        
        # Load replay buffer if requested
        if hasattr(args, "load_buffer") and args.load_buffer:
            model.load_replay_buffer(os.path.join(args.load_model_dir, "best_replay_buffer.zip"))
    else:
        # Create new model
        policy = load_policy(args)
        model = Agent(policy, env=vec_env, verbose=1, tensorboard_log=log_dir, **args.agent_kwargs)

    # Train model
    model.learn(total_timesteps=args.total_timesteps,
               progress_bar=True,
               callback=callback_list,
               tb_log_name=log_name,
               log_interval=100,
               reset_num_timesteps=False)

    # Save final model and environment
    model.save(os.path.join(checkpoint_dir, "final_model.zip"))
    vec_env.save(os.path.join(checkpoint_dir, 'final_env.zip'))

    # Save replay buffer if requested
    if getattr(args, "save_replaybuffer", False) and hasattr(model, 'save_replay_buffer'):
        model.save_replay_buffer(os.path.join(checkpoint_dir, 'final_replay_buffer.zip'))

def parse_args() -> tuple:
    """Parse command line arguments and load configuration file."""
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent")
    parser.add_argument('--config_file', '-f', type=str, default=None,
                       help="Path to configuration file")
    args = parser.parse_args()
    
    # Load and parse configuration
    config = json.load(open(args.config_file))
    with open(args.config_file) as f:
        config_str = f.read()
        
    arg_config = argparse.Namespace(**config)
    arg_config.total_config = config
    arg_config.config_name = args.config_file.split("/")[-1].split(".")[0]
    arg_config.config_file = args.config_file
    
    return arg_config, config_str

def main():
    args, config_str = parse_args()
    train(args, config_str)


if __name__ == "__main__":
    main()
