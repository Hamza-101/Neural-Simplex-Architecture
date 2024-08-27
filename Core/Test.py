import ray
from ray.rllib.algorithms.ppo import PPO
from env import FlockingEnv
from Settings import *

# Initialize Ray (if not already initialized)
ray.init(ignore_reinit_error=True)

# Assuming `flocking_env` is already registered and policies are set
# Load the trained MAPPO model
checkpoint_path = "/path/to/your/checkpoint"  # Update this with your actual checkpoint path
config = {
    "env": "flocking_env",
    "framework": "torch",
}

# Restore the PPO model from checkpoint
mappo_trainer = PPO(config=config)
mappo_trainer.restore(checkpoint_path)

# Load the environment
env = FlockingEnv()

# Number of episodes to run
num_episodes = 10

for episode in range(num_episodes):
    observations = env.reset()
    done = {"__all__": False}
    episode_reward = 0

    while not done["__all__"]:
        action_dict = {}
        
        # Compute actions for each agent
        for agent_id, obs in observations.items():
            action_dict[agent_id] = mappo_trainer.compute_single_action(observation=obs, policy_id=agent_id)

        # Apply actions to the environment
        observations, rewards, done, info = env.step(action_dict)

        # Accumulate rewards
        episode_reward += sum(rewards.values())

        # (Optional) Print the state of the environment, actions, or rewards

    print(f"Episode {episode + 1}: Total Reward: {episode_reward}")

# Shutdown Ray after testing
ray.shutdown()
