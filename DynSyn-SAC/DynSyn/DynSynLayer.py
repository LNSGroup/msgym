from typing import List
import torch as th
from torch import nn
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution

class DynSynLayer(nn.Module):
    """Layer that maps latent policy output to muscle activations via group weights."""

    def __init__(
        self,
        muscle_groups: List[List[int]],
        last_layer_dim: int,
        dynsyn_log_std: float,
    ) -> None:
        """Initialize the DynSyn layer.

        Args:
            muscle_groups: List of muscle index groups; each group shares one weight.
            last_layer_dim: Input dimension from the policy latent layer.
            dynsyn_log_std: Log std for the weight distribution.
        """
        super().__init__()
        self.muscle_groups = muscle_groups
        self.muscle_group_dims = [len(i) for i in muscle_groups]
        self.muscle_group_dims = th.tensor(self.muscle_group_dims)
        
        self.muscle_dims = max([max(i) for i in muscle_groups]) + 1
        self.muscle_group_nums = len(muscle_groups)

        self.weight_muscle_indx = []
        for i in self.muscle_groups:
            if len(i) > 1:
                self.weight_muscle_indx.extend(i[1:])
        self.weight_muscle_indx = th.tensor(self.weight_muscle_indx)

        # Calculate the index
        total_indx = 0
        self.muscle_indx = list(range(self.muscle_dims))
        for group in muscle_groups:
            for i in group:
                self.muscle_indx[i] = total_indx
                total_indx += 1
        self.muscle_indx = th.tensor(self.muscle_indx)

        self.mu = nn.Sequential(
            nn.Linear(last_layer_dim, self.muscle_dims - self.muscle_group_nums)
        )
        self.log_std = th.tensor(dynsyn_log_std)
        self.weight_dist = SquashedDiagGaussianDistribution(self.muscle_dims - self.muscle_group_nums)

        self.dynsyn_weight_amp = None

    def repeat_replace_x(self, x: th.Tensor) -> th.Tensor:
        """Repeat and reorder latent activations according to muscle group structure."""
        if x.device != self.muscle_group_dims.device:
            self.muscle_group_dims = self.muscle_group_dims.to(x.device)
            self.weight_muscle_indx = self.weight_muscle_indx.to(x.device)

        x = x.repeat_interleave(self.muscle_group_dims, dim=-1)
        x = x[..., self.muscle_indx]

        return x

    def forward(
        self,
        x: th.Tensor,
        latent_pi: th.Tensor,
        deterministic: bool = False,
    ) -> th.Tensor:
        """Apply synergistic weights to the action and clamp to [-1, 1]."""
        mean_weights = self.mu(latent_pi)
        log_std = self.log_std
        weight = self.weight_dist.actions_from_params(mean_weights, log_std, deterministic=deterministic)
        
        if self.dynsyn_weight_amp is not None:
            weight = th.clamp(weight * 0.1, -self.dynsyn_weight_amp, self.dynsyn_weight_amp) + 1
        else:
            weight = th.ones_like(weight)

        x = self.repeat_replace_x(x)
        x[..., self.weight_muscle_indx] *= weight
        x = th.clamp(x, -1, 1)

        return x

    def update_dynsyn_weight_amp(self, dynsyn_weight_amp: float) -> None:
        """Set the amplitude for clamping synergistic weights."""
        self.dynsyn_weight_amp = dynsyn_weight_amp
