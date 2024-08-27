import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from env import FlockingEnv
from Settings import *

# Config and kwargs not being used 

# # Initialize Ray
# ray.init()

# # Register the custom environment
# env_name = "flocking_env"
# tune.register_env(env_name, lambda config: FlockingEnv())

# # Define policies
# policy_ids = [f"agent_{i}" for i in range(3)]  # Assuming 3 agents

# # Configure MAPPO
# config = (
#     PPOConfig()
#     .environment(env_name)
#     .framework("torch")
#     .multi_agent(
#         policies={policy_id: (None, FlockingEnv().observation_space, FlockingEnv().action_space, {})
#                   for policy_id in policy_ids},
#         policy_mapping_fn=lambda agent_id, **kwargs: agent_id
#     )
#     .rollouts(num_rollout_workers=2)
#     .training(train_batch_size=4000, sgd_minibatch_size=64, num_sgd_iter=10)
# )

# # Train the MAPPO model
# tune.run(
#     "PPO",
#     config=config.to_dict(),
#     stop={"episode_reward_mean": 100},  # Adjust this based on your environment
# )

# # Shutdown Ray
# ray.shutdown()

# Initialize Ray

ray.init()

# Register the custom environment
env_name = "flocking_env"
tune.register_env(env_name, lambda config: FlockingEnv())

# Define policies
policy_ids = [f"agent_{i}" for i in range(3)]  # Assuming 3 agents

# Configure MAPPO with custom network architecture
config = PPOConfig() \
    .environment(env_name) \
    .framework("torch") \
    .multi_agent(
        policies={
            policy_id: (
                None,
                FlockingEnv().observation_space,
                FlockingEnv().action_space,
                {
                    "model": {
                        "vf_share_layers": True,  # Share layers between policy and critic
                        "fcnet_hiddens": [512, 512, 512, 512, 512, 512, 512, 512],  # Eight layers of 512 neurons each
                        "fcnet_activation": "tanh",  # Tanh activation function
                    },
                }
            )
            for policy_id in policy_ids
        },
        policy_mapping_fn=lambda agent_id, **kwargs: agent_id,
    ) \
    .rollouts(num_rollout_workers=2) \
    .training(train_batch_size=4000, sgd_minibatch_size=64, num_sgd_iter=10)

# Train the MAPPO model
tune.run(
    "PPO",
    config=config.to_dict(),  # Convert the config to a dictionary
    stop={"episode_reward_mean": 100},  # Adjust this based on your environment
)

# Shutdown Ray
ray.shutdown()