import ray
from ray.rllib.algorithms import ppo
from ray.rllib.env import MultiAgentEnv
from ray.tune.registry import register_env

# Make sure your environment class is imported
from env import FlockingEnv  # Replace with the actual module name

# Register your custom environment
def create_flocking_env(env_config):
    return FlockingEnv()

register_env("flocking_env", create_flocking_env)


# Initialize Ray
ray.init(ignore_reinit_error=True)

# Configuration for PPO
config = {
    "env": "flocking_env",
    "env_config": {},  # Custom environment configuration if needed
    "num_workers": 4,  # Number of parallel workers
    "num_envs_per_worker": 1,  # Number of environments per worker
    "model": {
        "fcnet_hiddens": [64, 64],  # Network architecture
    },
    "lr": 1e-4,  # Learning rate
    "train_batch_size": 4000,  # Total batch size for each update
    "rollout_fragment_length": 200,  # Length of each rollout fragment
    "framework": "tf",  # or "torch", depending on which framework you're using
}

# Create the PPO trainer
trainer = ppo.PPO(config=config, env="flocking_env")

# Training loop
for i in range(10000):  # Number of training iterations
    result = trainer.train()
    print(f"Iteration {i}: {result['episode_reward_mean']}")
    
    # Save checkpoint periodically
    if i % 1000 == 0:
        checkpoint = trainer.save()
        print(f"Checkpoint saved at {checkpoint}")

# Shutdown Ray
ray.shutdown()