import os
from typing import List, Optional, Tuple
import numpy as np

class LocomotionCycleTrajectory:
    """Loads and queries cyclic locomotion trajectories from .npz files."""

    def __init__(
        self,
        motion_dir: str,
        motion_list: Optional[List[int]] = None,
    ) -> None:
        """Initialize and load trajectories from disk.

        Args:
            motion_dir: Path to a single .npz file or a directory of .npz files.
            motion_list: Optional list of indices into the directory's .npz files
                to load. If None, all .npz files in motion_dir are loaded.
        """
        self.trajectories = self._load_all_trajectories(motion_dir, motion_list)
        self.num_trajectories = len(self.trajectories)

        if self.num_trajectories == 0:
            raise ValueError("No trajectories were loaded.")

    def _load_all_trajectories(
        self,
        motion_dir: str,
        motion_list: Optional[List[int]],
    ) -> List[dict]:
        """Load all specified trajectories from disk."""
        trajectories_data = []
        if os.path.isfile(motion_dir):
            trajectories_data.append(self._load_single_trajectory(motion_dir))
        # Auto-detect all .npz files if motion_dir is a directory and motion_list is not specified
        elif os.path.isdir(motion_dir):
            # Get all .npz files in the directory
            motion_file_list = [
                f for f in os.listdir(motion_dir)
                if f.endswith(".npz") and os.path.isfile(os.path.join(motion_dir, f))
            ]
            # Sort files alphabetically for consistent loading order
            motion_file_list.sort()
            print("All motion file list:", motion_file_list)
            if not motion_file_list:
                raise ValueError(f"No .npz files found in directory: {motion_dir}")
            if motion_list is not None:
                motion_file_list = [motion_file_list[i] for i in motion_list if 0 <= i < len(motion_file_list)]
                print("Selected motion file list:", motion_file_list)
            for motion_file in motion_file_list:
                full_path = os.path.join(motion_dir, motion_file)
                trajectories_data.append(self._load_single_trajectory(full_path))
            print(f"Auto-loaded {len(motion_file_list)} trajectory files from directory")
        else:
            raise ValueError(f"Invalid motion_dir or motion_list provided. motion_dir: {motion_dir}")
        return trajectories_data

    def _load_single_trajectory(self, file_path: str) -> dict:
        """Load one .npz trajectory, and precompute velocities."""
        try:
            data = np.load(file_path)
            qpos_traj = data['qpos_traj'].astype(np.float32)
            xpos_traj = data['xpos_traj'].astype(np.float32)

            num_frames = qpos_traj.shape[0]
            framerate = data['framerate']
            dt = 1.0 / framerate
            velocity = data['velocity']
            period = data['period']
            stride = data['stride']

            # Pre-compute qvel for faster querying
            qvel_traj = np.zeros_like(qpos_traj)
            # Calculate velocity for all but the last frame
            qvel_traj[:-1] = (qpos_traj[1:] - qpos_traj[:-1]) / dt
            # For the last frame, loop back to the first frame to ensure cycle continuity
            qvel_traj[-1] = (qpos_traj[0] - qpos_traj[-1]) / dt

            traj_data = {
                'qpos_traj': qpos_traj,
                'xpos_traj': xpos_traj,
                "qvel_traj": qvel_traj,
                'framerate': framerate,
                'velocity': velocity,
                'period': period,
                'stride': stride,
                'num_frames': num_frames,
                'dt': dt,
                'terminate_time': num_frames / framerate,
                'pelvis_trans': (xpos_traj[-1, 1, :] - xpos_traj[0, 1, :]).astype(np.float32)
            }
            return traj_data
        except FileNotFoundError:
            raise FileNotFoundError(f"Trajectory file not found at: {file_path}")
        except KeyError as e:
            raise KeyError(f"Trajectory file {file_path} is missing expected key: {e}")

    def get_trajectory_properties(
        self, traj_index: int
    ) -> Tuple[float, float, float]:
        """Return terminate_time, velocity, and stride for a trajectory."""
        if not 0 <= traj_index < self.num_trajectories:
            raise IndexError("Trajectory index out of range.")
        traj = self.trajectories[traj_index]
        return traj["terminate_time"], traj["velocity"], traj["stride"]

    def query(
        self, time: float, traj_index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Query qpos, xpos, and qvel at a given time for a trajectory.

        Args:
            time: Simulation time to query.
            traj_index: Index of the trajectory in the loaded set.

        Returns:
            Tuple of (qpos, xpos, qvel) at the queried time.
        """
        traj = self.trajectories[traj_index]
        terminate_time = traj["terminate_time"]
        framerate = traj["framerate"]
        num_frames = traj["num_frames"]

        cycle_number = int(np.floor(time / terminate_time))
        time_in_cycle = time % terminate_time
        time_step = int(np.round(time_in_cycle * framerate))
        time_step = np.clip(time_step, 0, num_frames - 1)

        qpos = traj["qpos_traj"][time_step].copy()
        xpos = traj["xpos_traj"][time_step].copy()
        qvel = traj["qvel_traj"][time_step].copy()

        # Apply cycle transformations for continuous motion
        if cycle_number > 0:
            qpos[0] += cycle_number * traj["qpos_traj"][-1, 0]
            qpos[2] += cycle_number * traj["qpos_traj"][-1, 2]
            xpos[1:, :2] += cycle_number * traj["pelvis_trans"][:2]

        return qpos, xpos, qvel

    def query_batch(
        self, times: np.ndarray, traj_index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized query for multiple times.

        Args:
            times: 1D array of times to query.
            traj_index: Index of the trajectory in the loaded set.

        Returns:
            Tuple of (qpos, xpos, qvel) with shapes (T, ...).
        """
        times = np.asarray(times)
        if times.ndim != 1:
            raise ValueError("times must be a 1D array")

        traj = self.trajectories[traj_index]
        terminate_time = traj["terminate_time"]
        framerate = traj["framerate"]
        num_frames = traj["num_frames"]

        cycle_number = np.floor(times / terminate_time).astype(np.int64)
        time_in_cycle = np.mod(times, terminate_time)
        time_step = np.rint(time_in_cycle * framerate).astype(np.int64)
        time_step = np.clip(time_step, 0, num_frames - 1)

        # Fancy indexing returns copies; that is fine and avoids per-time .copy() in Python.
        qpos = traj["qpos_traj"][time_step]
        xpos = traj["xpos_traj"][time_step]
        qvel = traj["qvel_traj"][time_step]

        if np.any(cycle_number > 0):
            cycle_f = cycle_number.astype(qpos.dtype, copy=False)
            qpos[:, 0] += cycle_f * traj["qpos_traj"][-1, 0]
            qpos[:, 2] += cycle_f * traj["qpos_traj"][-1, 2]
            pelvis_trans_xy = traj["pelvis_trans"][:2].astype(xpos.dtype, copy=False)
            xpos[:, 1:, :2] += cycle_f[:, None, None] * pelvis_trans_xy[None, None, :]

        return qpos, xpos, qvel
