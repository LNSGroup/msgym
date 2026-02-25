from gymnasium.envs.registration import register

register(
    id="msgym/ManipulationEnv-v1",
    entry_point="msgym.envs:ManipulationEnvV1",
    max_episode_steps=200,
)
register(
    id="msgym/LocomotionFullEnv-v1",
    entry_point="msgym.envs:LocomotionFullEnvV1",
    max_episode_steps=3000,
)
register(
    id="msgym/LocomotionLegsEnv-v1",
    entry_point="msgym.envs:LocomotionLegsEnvV1",
    max_episode_steps=3000,
)

__all__ = []