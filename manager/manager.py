'''
Manage the lifecycle of the game environment.
'''
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
from agent.agent import PossibleAgentDict
from buffer.buffer import Buffer
class Manager():
    def __init__(self, env_config, network_config, policy_configs, seed):
        self.env_config = env_config
        self.network_config = network_config
        self.policy_configs = policy_configs
        self.env : UnityParallelEnv = None
        self.file_name = self.env_config['file_name']
        self.no_graphics = self.env_config['no_graphics']
        self.worker_id = self.env_config['worker_id']
        self.possible_agent_dict:PossibleAgentDict
        self.seed = seed
    def make(self):
        if self.env != None:
            return self.env
        env = UnityEnvironment(
            file_name=self.file_name,
            no_graphics=self.no_graphics,
            worker_id=self.worker_id,
            seed=self.seed
        )
        self.env = UnityParallelEnv(env)
        self.possible_agent_dict = PossibleAgentDict(
            possible_agent_list=self.env.possible_agents,
        )
        return self.env
    def run(self, buffer : Buffer=None):
        reward_list = []
        stop = False
        obs = self.env.reset()
        dones = {}
        self.possible_agent_dict.update_group_info(self.env.infos)
        while not stop:
            if any([dones[a] for a in dones]):
                obs = self.env.reset()
            actions, logprobs, _, values, stop = buffer.update_actions(
                input_obs=obs,
                input_done=dones,
                possible_agent_dict=self.possible_agent_dict
            )
            obs, rewards, dones, infos = self.env.step(actions=actions)
            sum_rewards = buffer.update_rewards(
                input_reward=rewards,
                input_info=infos,
                possible_agent_dict=self.possible_agent_dict
            )
            reward_list.extend(sum_rewards)
            if stop:
                break
        buffer.calculate()
        buffer.flatten()
        return reward_list