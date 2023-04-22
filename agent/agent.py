from typing import Dict, List
import numpy as np
def get_observation_array(input_obs):
    obs_array = np.array([])
    if not isinstance(input_obs, dict):
        for ndarray in input_obs:
            obs_array = np.append(obs_array, ndarray)
    else:
        for ndarray in input_obs['observation']:
            obs_array = np.append(obs_array, ndarray)
    return obs_array
class PossibleAgentDict():
    def __init__(self, possible_agent_list):
        Agent = Dict[str, int]
        self.team_list:List[Agent] = []
        self.group_dict:Dict[str, str] = {}
        self.num_agents = len(possible_agent_list)
        self.num_teams = len(set(self._calculate_num_id(str_id,'team') for str_id in possible_agent_list))
        for i in range(self.num_teams):
            self.team_list.append({})
            team_list = list(filter(lambda str_id: self._calculate_num_id(str_id, 'team') == i, possible_agent_list))
            for str_id in team_list:
                self.team_list[i][str_id] = team_list.index(str_id)
        return
    def update_group_info(self, env_infos:dict):
        for str_id, agent_info in env_infos.items():
            self.group_dict[str_id] = agent_info['group_id']
        return
    def get_num_teams(self):
        return self.num_teams
    def get_group_id(self, str_id:str):
        return self.group_dict[str_id]
    def get_group_agents(self, group_id:int):
        group_agents = []
        for str_id, _group_id in self.group_dict.items():
            if group_id == _group_id:
                group_agents.append(str_id)
        return sorted(group_agents)
    def get_group_other_agents(self, str_id:str):
        group_agents = []
        group_id = self.get_group_id(str_id)
        for _str_id, _group_id in self.group_dict.items():
            if group_id == _group_id and str_id != _str_id:
                group_agents.append(str_id)
        return sorted(group_agents)
    def get_team_agents(self, team_id):
        return self.team_list[team_id]
    def get_num_id(self, str_id:str):
        team_id = self._calculate_num_id(str_id, 'team')
        return self.team_list[team_id][str_id]
    def _calculate_num_id(self, str_id:str, key_word:str):
        import re
        match = re.search(key_word + r'=(\d+)', str_id)
        if match:
            num_id = match[1]
        return int(num_id)