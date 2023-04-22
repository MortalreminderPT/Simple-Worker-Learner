from typing import List
import numpy as np
import torch
from buffer.buffer import Buffer
from util import get_observation_array
def batchify_obs(obs):
    m_obs = np.zeros(len(obs))
    obs = torch.tensor(m_obs)
    return obs
def get_num_id(str_id, key_words):
    import re
    match = re.search(key_words+r'=(\d+)', str_id)
    if match:
        num_id = match[1]
    return int(num_id)
class MAPPOBuffer(Buffer):
    def __init__(self, network_config, policy_config, policy_list : List, num_workers = 1, seed = 0):
        super().__init__(network_config, policy_config, seed)
        
        
        
        self.num_steps = self.network_config['num_steps'] // num_workers
        self.num_observations = self.policy_config['num_observations']
        self.algorithm = self.network_config['algorithm']
        if self.algorithm == 'ippo':
            self.global_num_observations = self.num_observations
        elif self.algorithm == 'mappo':
            self.global_num_observations = 2 * self.num_observations
        self.num_actions = self.policy_config['num_actions']
        self.buffer_size = self.network_config['batch_size'] // num_workers
        self.num_agents = self.network_config['num_agents']
        self.gamma = self.network_config['gamma']
        self.gae_lambda = self.network_config['gae_lambda']
        self.policys = policy_list
        self.obs = torch.zeros((self.num_steps, self.num_agents) + tuple([self.num_observations])).to(self.device)
        self.global_obs = torch.zeros((self.num_steps, self.num_agents, self.global_num_observations)).to(self.device)
        self.actions = torch.zeros(
            (self.num_steps, self.num_agents,) + tuple([self.num_actions[0]])
        ).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_agents)).to(self.device)
        self.logprobs = torch.zeros_like(self.rewards).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.num_agents)).to(self.device)
        self.values = torch.zeros((self.num_steps, self.num_agents)).to(self.device)
        self.advantages = torch.zeros((self.num_steps, self.num_agents)).to(self.device)
        
    def update_actions(self, input_obs:dict, input_done:dict, possible_agent_dict):
        with torch.no_grad():
            
            if len(input_done) < len(input_obs):
                for str_id in input_obs.keys():
                    if str_id not in input_done.keys():
                        input_done[str_id] = True
            action_return = {}
            for team_id in range(possible_agent_dict.get_num_teams()):
                obs_tensor = torch.zeros((self.num_agents, self.num_observations)).to(self.device)
                done_tensor =  torch.ones(self.num_agents).to(self.device)
                global_obs_tensor = torch.zeros((self.num_agents, self.global_num_observations)).to(self.device)
                for str_id in possible_agent_dict.get_team_agents(team_id):
                    agent_id:int = possible_agent_dict.get_num_id(str_id=str_id)
                    obs_tensor[agent_id] = torch.Tensor(get_observation_array(input_obs[str_id])).to(self.device)
                    done_tensor[agent_id] =  torch.as_tensor(input_done[str_id]).to(self.device)
                    global_obs_tensor[agent_id] = torch.Tensor(
                        np.append(
                            get_observation_array(input_obs=input_obs[str_id]),
                            [
                                get_observation_array(input_obs[other_str_id])
                                for other_str_id in possible_agent_dict.get_group_other_agents(str_id)
                            ]
                        ).flatten()
                        
                    ).to(self.device)
                action, logprob, _, value = self.policys[team_id].get_action_and_value(
                    input_obs=obs_tensor,
                    global_obs=global_obs_tensor,
                )
                
                if team_id == 0:
                    self.obs[self._step] = obs_tensor
                    self.global_obs[self._step] = global_obs_tensor
                    self.dones[self._step] = done_tensor
                    self.actions[self._step] = action
                    self.logprobs[self._step] = logprob
                    self.values[self._step] = torch.squeeze(value)
                for str_id in possible_agent_dict.get_team_agents(team_id):
                    agent_id:int = possible_agent_dict.get_num_id(str_id=str_id)
                    if input_done[str_id]:
                        continue
                    action_return[str_id] = torch.squeeze(action[agent_id].cpu())
            
            
            
            self._step += 1
            stop = self._step >= self.num_steps
            return action_return, logprob, None, value, stop
    def update_rewards(self, input_reward:dict, input_info:dict, possible_agent_dict):
        possible_agent_list = list(input_reward.keys() & possible_agent_dict.get_team_agents(team_id=0))
        sum_reward = []
        for str_id in possible_agent_list:
            agent_id = possible_agent_dict.get_num_id(str_id=str_id)
            
            sum_reward.append(input_reward[str_id] + input_info[str_id]['group_reward'])
            self.rewards[self._step - 1][agent_id] = torch.as_tensor(input_reward[str_id] + input_info[str_id]['group_reward'])
        return sum_reward
    def calculate(self):
        with torch.no_grad():
            next_done = self.dones[self._step - 1]
            next_value = self.policys[0].get_value(
                global_obs=self.global_obs[self._step - 1]
            ).reshape(1, -1)
            last_gae_lam = 0
            for step in reversed(range(self._step)):
                if step == self._step - 1:
                    next_non_terminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - self.dones[step + 1]
                    next_values = self.values[step + 1]
                delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
                self.advantages[step] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.returns = self.advantages + self.values
    def flatten(self):
        self.obs_flatten = self.obs.reshape((-1,) + tuple([self.num_observations]))
        self.global_obs_flatten = self.global_obs.reshape((-1,) + tuple([self.global_num_observations]))
        self.actions_flatten = self.actions.reshape((-1,self.num_actions[0]))
        self.logprobs_flatten = self.logprobs.reshape(-1)
        self.advantages_flatten = self.advantages.reshape(-1)
        self.returns_flatten = self.returns.reshape(-1)
        self.values_flatten = self.values.reshape(-1)
        self.dones_flatten = self.dones.reshape(-1)