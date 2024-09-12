import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
# from ray.rllib.algorithms.ppo import PPO
from env import FlockingEnv
from Settings import *
from ray.rllib.algorithms.callbacks import DefaultCallbacks


# Config and kwargs not being used 

class LossCallback(DefaultCallbacks):
    def __init__(self, loss_threshold=2000):
        super().__init__()
        self.loss_threshold = loss_threshold

    def on_train_result(self, *, trainer, result, **kwargs):
        # Assuming "policy_loss" or similar is in the result dict
        # Adjust this key according to your setup
        if 'policy_loss' in result['info']:
            policy_loss = result['info']['policy_loss']
            # Stopping if the average policy loss is below threshold
            if policy_loss < self.loss_threshold:
                print(f"Stopping training as loss ({policy_loss}) is below {self.loss_threshold}")
                result['done'] = True  # This will signal to stop the training


ray.init()

# Register the custom environment
env_name = "flocking_env"
tune.register_env(env_name, lambda config: FlockingEnv())

# Define policies
policy_ids = [f"agent_{i}" for i in range(3)]  # Assuming 3 agents

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
    .training(train_batch_size=2048, sgd_minibatch_size=64, num_sgd_iter=10)

tune.run(
    "PPO",
    config=config.to_dict(),  # Convert the config to a dictionary
    stop={"episode_reward_mean": 100},  # Adjust this based on your environment
)

# tune.run(
#     PPO,
#     config=config,  # No need to convert to dict
#     stop={"episode_reward_mean": 100},  # Adjust this based on your environment
# )

# tune.run(
#     PPO,
#     config=config,
#     stop={"training_iteration": 100},  # Optional stop condition if loss threshold isn't hit
#     callbacks=[LossCallback(loss_threshold=2000)]
# )

# Shutdown Ray
ray.shutdown()