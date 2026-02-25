from typing import Any, Union
import gymnasium as gym
import numpy as np

class MuscleNormWrapper(gym.ActionWrapper):
    """Maps actions from [-1, 1] to [0, 1] using a sigmoid for muscle activation."""
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.env.action_space.shape[0],)
        )
        
    def action(self, action: Union[np.ndarray, Any]) -> np.ndarray:
        action = 1.0 / (1.0 + np.exp(-5.0 * (action - 0.5)))
        return action