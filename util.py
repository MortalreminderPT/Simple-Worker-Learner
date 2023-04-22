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
def get_agent_id(agent_id:str):
    import re
    match = re.search(r'agent_id=(\d+)', agent_id)
    if match:
        agent_id = match[1]
    return int(agent_id)
