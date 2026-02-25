import os
import sys
import gymnasium as gym
import msgym

def main() -> None:
    env = gym.make("msgym/LocomotionFullEnv-v1", render_mode='human', gait_cycles=5,
                   kinematic_play=True, random_init=True)

    # env = gym.make("msgym/LocomotionLegsEnv-v1", render_mode='human', gait_cycles=5,
    #                kinematic_play=True, random_init=True)

    # env = gym.make("msgym/ManipulationEnv-v1", render_mode='human')
    
    episodes = 10
    for episode in range(episodes):
        terminated = truncated = False
        obs, info = env.reset()
        print(f"\nEpisode {episode + 1}/{episodes}")
        print("Action space shape:", env.action_space.shape)
        print("Observation space shape:", obs.shape)

        while not terminated and not truncated:
            # Use zero action for visualization
            action = env.action_space.sample() * 0
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            print("Reward:", info)

    env.close()

if __name__ == "__main__":
    main()
