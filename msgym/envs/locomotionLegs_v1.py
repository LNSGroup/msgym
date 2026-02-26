import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import mujoco
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils import EzPickle, seeding
from msgym.envs.imitation_trajectory import LocomotionCycleTrajectory
from msgym.envs.utils import action_obs_check, get_render_fps

class LocomotionLegsEnvV1(MujocoEnv, EzPickle):
    """Legs-only locomotion imitation: track reference motion from .npz trajectory (single or multiple)."""
    metadata: Dict[str, Any] = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "w_qpos": 50, 
        "w_xpos": 50,
        "w_pelvis": 50,  # Weight for pelvis position tracking
        "w_energy": 0.1, 
        "w_healthy": 100,
    }

    def __init__(
        self,
        motion_dir: Optional[str] = None,
        motion_list: Optional[List[int]] = None,
        gait_cycles: int = 5,
        skip_frames: int = 10,
        reset_noise_scale: float = 1e-3,
        random_init: bool = True,
        qpos_diff_th: float = 0.05,
        kinematic_play: bool = False,
        render_mode: Optional[str] = None,
        reward_dict: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the legs-only locomotion imitation environment.

        Args:
            motion_dir: Path to a single .npz file or directory of .npz files; None uses default.
            motion_list: Indices of .npz files to load from motion_dir (0-based).
            gait_cycles: Number of gait cycles before episode ends.
            skip_frames: Simulation steps per environment step.
            reset_noise_scale: Scale of noise added to qpos/qvel on reset.
            random_init: If True, start at a random time in the trajectory.
            qpos_diff_th: Threshold on mean qpos error for healthy state.
            kinematic_play: If True, gravity off and state follows reference.
            render_mode: One of "human", "rgb_array", "depth_array", or None.
            reward_dict: Reward weights; None uses DEFAULT_RWD_KEYS_AND_WEIGHTS.
            **kwargs: Passed to MujocoEnv.
        """
        if reward_dict is None:
            reward_dict = self.DEFAULT_RWD_KEYS_AND_WEIGHTS
        model_path = os.path.join(os.path.dirname(__file__), "..", "..","MS-Human-700", "MS-Human-700-Locomotion.xml")
        model_path = os.path.abspath(model_path)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        fps = get_render_fps(model_path, skip_frames)
        self.metadata["render_fps"] = fps
        self.control_timestep = 1 / fps

        if motion_dir is None:
            # Default behavior for backward compatibility
            motion_dir = os.path.join(os.path.dirname(__file__), "..", "motion_data", "walking_gait.npz")
            motion_dir = os.path.abspath(motion_dir)
            motion_list = None

        # Initialize trajectory manager with shared memory store
        self.trajectory = LocomotionCycleTrajectory(motion_dir, motion_list)
        # Track currently selected trajectory index
        self.current_traj_index = 0
        self.future_traj_steps = 5

        self.render_mode = render_mode
        self.reward_weight = reward_dict
        self._reset_noise_scale = reset_noise_scale
        self.cycles = gait_cycles
        # Get initial trajectory properties using the trajectory API
        terminate_time, velocity, stride = self.trajectory.get_trajectory_properties(self.current_traj_index)
        self.terminate_time = terminate_time
        self.random_init = random_init
        self.kinematic_play = kinematic_play
        self.qpos_diff_th = qpos_diff_th

        self.key_body_names = ["sternum", "head_neck", 
                               "toes_r", "toes_l"]

        EzPickle.__init__(
            self,
            motion_dir,
            motion_list,
            render_mode,
            skip_frames,
            reset_noise_scale,
            **kwargs,
        )

        # Dummy observation space to initialize MujocoEnv
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        MujocoEnv.__init__(
            self, model_path, skip_frames, observation_space=observation_space, render_mode=render_mode, camera_name="record_camera", max_geom=10000, **kwargs
        )

        self.init_qpos[:] = self.model.key_qpos[0].copy()
        # Initialize body indices
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.key_body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.key_body_names]
        self.body_name_list = [self.model.body(body_id).name for body_id in range(self.model.nbody)]
        self.joint_name_list = [self.model.joint(jnt_id).name for jnt_id in range(self.model.njnt)]
        self.muscle_name_list = [self.model.actuator(act_id).name for act_id in range(self.model.nu)]

        self._get_model_mapping()
        
        # Get initial reference state (current + future) in one batched query
        init_times = self.data.time + np.arange(self.future_traj_steps + 1, dtype=np.float64) * self.dt
        qpos_batch_full, xpos_batch_full, _ = self.trajectory.query_batch(init_times, self.current_traj_index)
        self.qpos_ref_full = qpos_batch_full[0]
        self.xpos_ref_full = xpos_batch_full[0]
        self.qpos_ref = self.qpos_ref_full[self.joint_map]
        self.xpos_ref = self.xpos_ref_full[self.body_map]
        self.qpos_ref_future = np.zeros((self.future_traj_steps, self.model.nq))
        self.qpos_ref_future[:, :] = qpos_batch_full[1 : 1 + self.future_traj_steps][:, self.joint_map]
        
        # Set the actual observation space
        obs, _ = self._get_obs()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)
        action_obs_check(self)

        if self.kinematic_play:
            self.model.opt.gravity[2] = 0.0
        print("Observation space shape:", self.observation_space.shape)
        print("Action space shape:", self.action_space.shape)
        print(f"Loaded {self.trajectory.num_trajectories} trajectories.")

    def seed(self, seed: int = 0) -> list[int]:
        """Compatibility API for callers that still use env.seed(seed)."""
        self._np_random, seeded = seeding.np_random(seed)
        self._np_random_seed = seeded
        return [seeded]

    @property
    def is_healthy(self) -> bool:
        """Check if the model is in a healthy state (e.g., not fallen over)."""
        qpos_diff = np.abs(self.data.qpos[3:] - self.qpos_ref[3:]).mean()
        if qpos_diff > self.qpos_diff_th:
            return False
        return True

    @property
    def terminated(self) -> bool:
        """True if episode is done (unhealthy or max cycles reached)."""
        terminated = not self.is_healthy
        if self.data.time >= self.terminate_time * self.cycles:
            terminated = True
        return terminated

    def _get_obs(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Build observation vector and dict of components."""
        # Current state
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qacc = self.data.qacc.copy()
        act = self.data.act.flat.copy()
        actuator_forces = self.data.actuator_force.flat.copy() / 1000
        actuator_forces = actuator_forces.clip(-100, 100)
        actuator_length = self.data.actuator_length.flat.copy()
        actuator_velocity = self.data.actuator_velocity.flat.copy().clip(-100, 100)

        # Get pelvis position
        pelvis_xpos = self.data.xpos[self.pelvis_id].copy()
        pelvis_xpos_ref = self.xpos_ref[self.pelvis_id].copy()
        
        # Calculate relative positions for key bodies
        key_xpos = self.data.xpos[self.key_body_ids].copy()
        key_xpos = key_xpos - pelvis_xpos  # Convert to relative positions
        self.key_xpos_ref = self.xpos_ref[self.key_body_ids].copy()
        self.key_xpos_ref = self.key_xpos_ref - pelvis_xpos_ref  # Convert reference to relative positions
        key_xpos = key_xpos.flat.copy()
        self.key_xpos_ref = self.key_xpos_ref.flat.copy()

        obs_dict = {
            "qpos": qpos,
            "qvel": qvel,
            "qacc": qacc,
            "act": act,
            "actuator_forces": actuator_forces,
            "actuator_length": actuator_length,
            "actuator_velocity": actuator_velocity,
            "key_xpos": key_xpos,
            "qpos_ref": self.qpos_ref,
            "qpos_ref_future": self.qpos_ref_future.flat.copy(),
            "key_xpos_ref": self.key_xpos_ref,
        }
        
        observation = np.concatenate([obs_dict[key] for key in obs_dict.keys()]).astype(np.float32, copy=False)
        return observation, obs_dict

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step the environment with the given action."""
        if self.kinematic_play:
            self.data.qpos[:] = self.qpos_ref
            self.data.qvel[:] = 0
            self.data.qacc[:] = 0
            action = action * 0
        
        self.do_simulation(action, self.frame_skip)

        # Get current and reference pelvis positions
        pelvis_xpos = self.data.xpos[self.pelvis_id]
        pelvis_xpos_ref = self.xpos_ref[self.pelvis_id]
        
        qpos_reward = self._get_qpos_reward(self.data.qpos, self.qpos_ref) * self.reward_weight["w_qpos"]
        xpos_reward = self._get_xpos_reward(self.data.xpos[self.key_body_ids] - pelvis_xpos, 
                                          self.xpos_ref[self.key_body_ids] - pelvis_xpos_ref) * self.reward_weight["w_xpos"]
        pelvis_reward = self._get_pelvis_reward(pelvis_xpos, pelvis_xpos_ref) * self.reward_weight["w_pelvis"]
        energy_reward = self._get_energy_reward() * self.reward_weight["w_energy"]
        healthy_reward = self._get_healthy_reward() * self.reward_weight["w_healthy"]
        
        reward = qpos_reward + xpos_reward + pelvis_reward + energy_reward + healthy_reward

        terminated = self.terminated
        truncated = self.data.time >= self.terminate_time * self.cycles

        # Update reference state (current + future) with a single batched query
        ref_time = self.data.time + self.init_time + self.dt
        ref_times = ref_time + np.arange(self.future_traj_steps + 1, dtype=np.float64) * self.dt
        qpos_batch_full, xpos_batch_full, _ = self.trajectory.query_batch(ref_times, self.current_traj_index)
        self.qpos_ref_full = qpos_batch_full[0]
        self.xpos_ref_full = xpos_batch_full[0]
        self.qpos_ref = self.qpos_ref_full[self.joint_map]
        self.xpos_ref = self.xpos_ref_full[self.body_map]
        self.qpos_ref_future[:, :] = qpos_batch_full[1 : 1 + self.future_traj_steps][:, self.joint_map]
        observation, obs_dict = self._get_obs()

        info = {
            "reward_qpos": qpos_reward,
            "reward_xpos": xpos_reward,
            "reward_pelvis": pelvis_reward,
            "reward_energy": energy_reward,
            "reward_healthy": healthy_reward,
            "total_reward": reward,
        }

        return observation, reward, terminated, truncated, info

    def reset_model(self) -> np.ndarray:
        """Reset the model to a (possibly random) state from the trajectory."""
        
        # If multiple trajectories, select one at random
        if self.trajectory.num_trajectories > 1:
            if self.kinematic_play:
                self.current_traj_index = (self.current_traj_index + 1) % self.trajectory.num_trajectories
                terminate_time, velocity, stride = self.trajectory.get_trajectory_properties(self.current_traj_index)
                print(f'traj index: {self.current_traj_index}, traj vel: {velocity} m/s, traj stride: {stride} m')
            else:
                self.current_traj_index = self.np_random.integers(0, self.trajectory.num_trajectories)

        terminate_time, velocity, stride = self.trajectory.get_trajectory_properties(self.current_traj_index)
        self.terminate_time = terminate_time

        if self.random_init:
            # Randomly sample a time from the trajectory
            self.init_time = self.np_random.integers(0, int(self.terminate_time // self.dt)) * self.dt
        else:
            self.init_time = 0.0
            
        # Get initial state (current + future) from trajectory with current index
        ref_times = self.init_time + np.arange(self.future_traj_steps + 1, dtype=np.float64) * self.dt
        qpos_batch_full, xpos_batch_full, qvel_batch_full = self.trajectory.query_batch(ref_times, self.current_traj_index)
        self.qpos_ref_full = qpos_batch_full[0]
        self.xpos_ref_full = xpos_batch_full[0]
        self.qpos_ref = self.qpos_ref_full[self.joint_map]
        self.xpos_ref = self.xpos_ref_full[self.body_map]
        init_qvel = qvel_batch_full[0][self.joint_map]
        self.qpos_ref_future[:, :] = qpos_batch_full[1 : 1 + self.future_traj_steps][:, self.joint_map]

        # Add noise to the initial state
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        init_qpos = self.qpos_ref + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        init_qvel += self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(init_qpos, init_qvel)

        observation, _ = self._get_obs()
        return observation

    def _get_qpos_reward(
        self, qpos_real: np.ndarray, qpos_ref: np.ndarray
    ) -> float:
        """Reward for matching reference joint positions (excluding pelvis translation)."""
        return -np.sum(np.square(qpos_real[3:] - qpos_ref[3:]))

    def _get_xpos_reward(
        self, xpos_real: np.ndarray, xpos_ref: np.ndarray
    ) -> float:
        """Reward for matching reference body positions."""
        return -np.sum(np.square(xpos_real - xpos_ref))

    def _get_energy_reward(self) -> float:
        """Penalty for high muscle force (energy usage)."""
        return np.sum(self.data.actuator_force) / self.model.na

    def _get_healthy_reward(self) -> float:
        """Reward for staying healthy (upright)."""
        return 1.0 if self.is_healthy else 0.0

    def _get_pelvis_reward(
        self, pelvis_pos: np.ndarray, pelvis_pos_ref: np.ndarray
    ) -> float:
        """Reward for matching reference pelvis position."""
        return -np.sum(np.square(pelvis_pos - pelvis_pos_ref))

    def _get_model_mapping(self) -> None:
        """Build joint/body index mappings between full and reduced models."""
        full_model_path = os.path.join(os.path.dirname(__file__), "..", "..", "MS-Human-700", "MS-Human-700.xml")
        full_model_path = os.path.abspath(full_model_path)
        full_model = mujoco.MjModel.from_xml_path(full_model_path)
        full_body_name_list = [full_model.body(body_id).name for body_id in range(full_model.nbody)]
        full_joint_name_list = [full_model.joint(jnt_id).name for jnt_id in range(full_model.njnt)]

        self.joint_map = np.zeros(self.model.nq)
        self.body_map = np.zeros(self.model.nbody)
        for i in range(self.model.nq):
            self.joint_map[i] = full_joint_name_list.index(self.joint_name_list[i])
        for i in range(self.model.nbody):
            self.body_map[i] = full_body_name_list.index(self.body_name_list[i])

        self.joint_map = self.joint_map.astype(int)
        self.body_map = self.body_map.astype(int)