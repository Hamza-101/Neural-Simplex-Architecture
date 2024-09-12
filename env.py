import json
import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from Settings import *

class Agent:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.acceleration = np.zeros(2)
        self.max_acceleration = SimulationVariables["AccelerationUpperLimit"]
        self.velocity = np.round(np.random.uniform(-SimulationVariables["VelocityUpperLimit"], SimulationVariables["VelocityUpperLimit"], size=2), 2)
        self.max_velocity = SimulationVariables["VelocityUpperLimit"]

    def update(self, action):
        self.acceleration += action
        self.acceleration = np.clip(self.acceleration, -self.max_acceleration, self.max_acceleration)
        self.velocity += self.acceleration * SimulationVariables["dt"]
        vel = np.linalg.norm(self.velocity)
        if vel > self.max_velocity:
            self.velocity = (self.velocity / vel) * self.max_velocity
        self.position += self.velocity * SimulationVariables["dt"]
        return self.position, self.velocity

class FlockingEnv(MultiAgentEnv):
    def __init__(self):
        super(FlockingEnv, self).__init__()
        self.episode=0
        self.counter=1
        self.CTDE=False
        self.current_timestep = 0
        self.reward_log = []

        self.agents = [Agent(position) for position in self.read_agent_locations()]

        # Per agent action and observation space
        self.action_space = spaces.Box(low=np.array([-5, -5], dtype=np.float32),
                                       high=np.array([5, 5], dtype=np.float32),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-np.inf, -np.inf, -2.5, -2.5], dtype=np.float32),
                                            high=np.array([np.inf, np.inf, 2.5, 2.5], dtype=np.float32),
                                            dtype=np.float32)


    # def step(self, action_dict):
    #     observations = {}
    #     rewards = {}
    #     dones = {"__all__": False}
    #     infos = {}

    #     # Apply noisy actions
    #     for agent_id, action in action_dict.items():
    #         noisy_action = action + np.random.normal(loc=0, scale=0.5, size=action.shape)
    #         action = np.clip(noisy_action, self.action_space.low, self.action_space.high)
    #         position, velocity = self.agents[agent_id].update(action)

    #         observations[agent_id] = np.concatenate([position, velocity])

    #         # Calculate reward and check collisions
    #         reward, out_of_flock = self.calculate_reward(self.agents[agent_id])
    #         rewards[agent_id] = reward

    #         if self.check_collision(self.agents[agent_id]) or out_of_flock:
    #             dones["__all__"] = True  # End episode if a collision occurs
    #             self.reset()

    #     self.current_timestep += 1

    #     return observations, rewards, dones, infos

    def step(self, action_dict):
        observations = {}
        rewards = {}
        dones = {"__all__": False}
        infos = {}

        # Apply noisy actions and update each agent's state
        for agent_id, action in action_dict.items():
            # Apply noise to actions and update agent's state
            noisy_action = action + np.random.normal(loc=0, scale=0.5, size=action.shape)
            action = np.clip(noisy_action, self.agent_action_space.low, self.agent_action_space.high)

            # Update agent position and velocity
            position, velocity = self.agents[agent_id].update(action)

            # Update observation for each agent
            observations[agent_id] = np.concatenate([position, velocity])

            # Calculate reward
            reward, out_of_flock = self.calculate_reward(self.agents[agent_id])
            rewards[agent_id] = reward

            # If there's a collision or agent is out of flock, mark the episode as done
            if self.check_collision(self.agents[agent_id]) or out_of_flock:
                dones["__all__"] = True  # End episode if a collision occurs
                # Reset environment for all agents
                self.reset()

        # Update the time step count
        self.current_timestep += 1

        return observations, rewards, dones, infos

    # def reset(self):
    #     self.agents = {f"agent_{i}": Agent(position) for i, position in enumerate(self.read_agent_locations())}
        
    #     observations = {}
    #     for agent_id, agent in self.agents.items():
    #         agent.acceleration = np.zeros(2)
    #         agent.velocity = np.round(np.random.uniform(-SimulationVariables["VelocityUpperLimit"], SimulationVariables["VelocityUpperLimit"], size=2), 2)
    #         observations[agent_id] = np.concatenate([agent.position, agent.velocity])

    #     return observations

    def reset(self):
        # Reset all agents
        self.agents = {f"agent_{i}": Agent(position) for i, position in enumerate(self.read_agent_locations())}
        
        # Initialize observations for each agent
        observations = {}
        for agent_id, agent in self.agents.items():
            agent.acceleration = np.zeros(2)
            agent.velocity = np.round(np.random.uniform(-SimulationVariables["VelocityUpperLimit"], SimulationVariables["VelocityUpperLimit"], size=2), 2)
            observations[agent_id] = np.concatenate([agent.position, agent.velocity])

        return observations

    def close(self):
        print("Environment is closed. Cleanup complete.")
        
    def simulate_agents(self, actions):
        observations = []  # Initialize an empty 1D array

        actions_reshaped = actions.reshape(((SimulationVariables["SimAgents"]), 2))

        for i, agent in enumerate(self.agents):
            position, velocity = agent.update(actions_reshaped[i])
            observation_pair = np.concatenate([position, velocity])
            observations = np.concatenate([observations, observation_pair])  # Concatenate each pair directly

        return observations
    
    def check_collision(self, agent):

        for other in self.agents:
            if agent != other:
                distance = np.linalg.norm(agent.position - other.position)
                if distance < SimulationVariables["SafetyRadius"]:
                    return True
                
        return False

    def get_observation(self):
        observations = np.zeros((len(self.agents), 4), dtype=np.float32)
        
        for i, agent in enumerate(self.agents):
            observations[i] = [
                agent.position[0],
                agent.position[1],
                agent.velocity[0],
                agent.velocity[1]
            ]

        # Reshape the observation into 1D                    
        return observations
   
    def get_closest_neighbors(self, agent):

        neighbor_positions=[]
        neighbor_velocities=[]

        for _, other in enumerate(self.agents):
            if agent != other:
                distance = np.linalg.norm(other.position - agent.position)

                if(self.CTDE == True):

                    ################################################################
                    # if distance < SimulationVariables["NeighborhoodRadius"]:
                    #    neighbor_positions.append(other.position)
                    #    neighbor_velocities.append(other.velocity)
                    ################################################################
                    neighbor_positions.append(other.position)
                    neighbor_velocities.append(other.velocity)

                else:  
                    neighbor_positions.append(other.position)
                    neighbor_velocities.append(other.velocity)

        return neighbor_positions, neighbor_velocities         
   
    # def calculate_reward(self):
    #     reward=0
    #     Collisions={}

    #     neighbor_positions = [[] for _ in range(len(self.agents))] 
    #     neighbor_velocities = [[] for _ in range(len(self.agents))]
    #     out_of_flock=False

    #     for idx, _ in enumerate(self.agents):
    #         Collisions[idx] = []

    #     for _, agent in enumerate(self.agents): 
    #         neighbor_positions, neighbor_velocities=self.get_closest_neighbors(agent)
    #         val, out_of_flock=self.reward(agent, neighbor_velocities, neighbor_positions)
    #         reward+=val

    #     return reward, out_of_flock

    def calculate_reward(self, agent):
        reward = 0
        out_of_flock = False

        # Get closest neighbors for a single agent
        neighbor_positions, neighbor_velocities = self.get_closest_neighbors(agent)

        # Calculate reward for the single agent
        reward, out_of_flock = self.reward(agent, neighbor_velocities, neighbor_positions)

        return reward, out_of_flock

    def reward(self, agent, neighbor_velocities, neighbor_positions):
        CohesionReward = 0
        AlignmentReward = 0
        total_reward = 0
        outofflock = False
        midpoint = (SimulationVariables["SafetyRadius"] + SimulationVariables["NeighborhoodRadius"]) / 2

        if len(neighbor_positions) > 0:
            for neighbor_position in neighbor_positions:
                distance = np.linalg.norm(agent.position - neighbor_position)

                if distance <= SimulationVariables["SafetyRadius"]:
                    CohesionReward += -10
                elif SimulationVariables["SafetyRadius"] < distance <= midpoint:
                    CohesionReward += (10 / (midpoint - SimulationVariables["SafetyRadius"])) * (distance - SimulationVariables["SafetyRadius"])
                elif midpoint < distance <= SimulationVariables["NeighborhoodRadius"]:
                    CohesionReward += 10 - (10 / (SimulationVariables["NeighborhoodRadius"] - midpoint)) * (distance - midpoint)
      
                average_velocity = np.mean(neighbor_velocities, axis=0)
                dot_product = np.dot(average_velocity, agent.velocity)
                norm_product = np.linalg.norm(average_velocity) * np.linalg.norm(agent.velocity)

                if norm_product == 0:
                    cos_angle = 1.0
                else:
                    cos_angle = dot_product / norm_product

                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                orientation_diff = np.arccos(cos_angle)

                alignment = (orientation_diff / np.pi)
                AlignmentReward = -20 * alignment + 10  

        else:
            CohesionReward -= 10
            outofflock = True

        total_reward = CohesionReward + AlignmentReward

        return total_reward, outofflock

    def read_agent_locations(self):

        File = rf"{Results['InitPositions']}" + str(self.counter) + "\config.json"
        with open(File, "r") as f:
            data = json.load(f)

        return data