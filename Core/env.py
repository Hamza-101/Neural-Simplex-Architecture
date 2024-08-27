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
        self.agents = {f"agent_{i}": Agent(position) for i, position in enumerate(self.read_agent_locations())}
        self.CTDE = False
        self.current_timestep = 0
        self.reward_log = []

        # Action and observation spaces for each agent
        min_action = np.array([-5, -5], dtype=np.float32)
        max_action = np.array([5, 5], dtype=np.float32)
        min_obs = np.array([-np.inf, -np.inf, -2.5, -2.5], dtype=np.float32)
        max_obs = np.array([np.inf, np.inf, 2.5, 2.5], dtype=np.float32)

        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)
        self.observation_space = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def step(self, action_dict):
        observations = {}
        rewards = {}
        done = {}
        info = {}

        for agent_id, action in action_dict.items():
            noisy_action = action + np.random.normal(loc=0, scale=0.5, size=action.shape)
            clipped_action = np.clip(noisy_action, self.action_space.low, self.action_space.high)
            self.agents[agent_id].update(clipped_action)

            # Observe new state and calculate reward
            observations[agent_id] = self.get_observation(agent_id)
            rewards[agent_id], out_of_flock = self.calculate_reward(agent_id)

            if self.check_collision(self.agents[agent_id]) or out_of_flock:
                done[agent_id] = True
                done["__all__"] = True
            else:
                done[agent_id] = False

        done["__all__"] = all(done.values())
        return observations, rewards, done, info

    def reset(self):
        self.agents = {f"agent_{i}": Agent(position) for i, position in enumerate(self.read_agent_locations())}
        return {agent_id: self.get_observation(agent_id) for agent_id in self.agents}

    def get_observation(self, agent_id):
        agent = self.agents[agent_id]
        return np.array([agent.position[0], agent.position[1], agent.velocity[0], agent.velocity[1]])

    def calculate_reward(self, agent_id):
        agent = self.agents[agent_id]
        neighbor_positions, neighbor_velocities = self.get_closest_neighbors(agent)
        total_reward, out_of_flock = self.reward(agent, neighbor_velocities, neighbor_positions)
        return total_reward, out_of_flock

    def check_collision(self, agent):
        for other in self.agents.values():
            if agent != other:
                distance = np.linalg.norm(agent.position - other.position)
                if distance < SimulationVariables["SafetyRadius"]:
                    return True
        return False

    def get_closest_neighbors(self, agent):
        neighbor_positions = []
        neighbor_velocities = []

        for other in self.agents.values():
            if agent != other:
                distance = np.linalg.norm(other.position - agent.position)
                if distance < SimulationVariables["NeighborhoodRadius"]:
                    neighbor_positions.append(other.position)
                    neighbor_velocities.append(other.velocity)
                    
        return neighbor_positions, neighbor_velocities

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
        file = rf"{Results['InitPositions']}" + str(self.counter) + "\config.json"
        with open(file, "r") as f:
            data = json.load(f)
        return data
