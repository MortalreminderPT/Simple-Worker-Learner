'''
PPO with LSTM policy.
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
        self.policy_id = policy_config['policy_id']
        self.hidden_units = network_config['hidden_units']
        self.lstm = nn.LSTM(self.num_observations, self.hidden_units)
        self.actor =nn.Sequential(
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_units, self.hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_units, self.hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_units, np.prod(self.num_actions)), std=0.01),
        )
        self.critic = nn.Sequential(
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_units, self.hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_units, self.hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_units, 1), std=1.0),
        )
    def get_states(self, x, lstm_state, done):
        hidden = x
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state
    def get_value(self, input_obs, lstm_state, input_done):
        hidden, _ = self.get_states(input_obs, lstm_state, input_done)
        return self.critic(hidden)
    def get_action_and_value(self, input_obs, lstm_state, input_done, action = None):
        hidden, lstm_state = self.get_states(input_obs, lstm_state, input_done)
        logits = self.actor(hidden)
        logits = torch.reshape(logits, (input_obs.shape[0],)+tuple(self.num_actions))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(hidden), lstm_state