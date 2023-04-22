'''
PPO policy.
'''
from typing import Dict, Any, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
class Policy(nn.Module):
    def __init__(
            self,
            network_config: Dict[str, Any],
            policy_config: Dict[str, Any]
    ):
        super().__init__()
        self.num_observations =policy_config['num_observations']
        self.num_actions = policy_config['num_actions']
        self.global_state_size = self.num_observations
        self.policy_id = policy_config['policy_id']
        self.hidden_units = network_config['hidden_units']
        self.actor =nn.Sequential(
            layer_init(nn.Linear(self.num_observations, self.hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_units, self.hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_units, self.hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_units, np.prod(self.num_actions)), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.global_state_size, self.hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_units, self.hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_units, self.hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_units, 1), std=1.0),
        )

    def get_value(self, global_obs):
        if isinstance(global_obs, list):
            obs_tensor = torch.zeros(global_obs[0].shape[0], self.num_observations * len(global_obs))
            for i in range(len(global_obs)):
                obs = global_obs[i]
                if isinstance(obs, tuple):
                    grid_obs = self.grid(obs[0])
                    vector_obs = obs[1]
                    obs = torch.cat(grid_obs, vector_obs)
                obs_tensor[i] = obs
            return self.critic(obs_tensor)
        else:
            return self.critic(global_obs)

    def get_action_and_value(self, input_obs, global_obs = None, action = None):
        if isinstance(input_obs, tuple):
            grid_obs = self.grid(input_obs[0])
            vector_obs = input_obs[1]
            input_obs = torch.cat(grid_obs, vector_obs)
        value = None if global_obs is None else self.critic(global_obs)
        logits = self.actor(input_obs)
        logits = torch.reshape(logits, (input_obs.shape[0],)+tuple(self.num_actions))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value