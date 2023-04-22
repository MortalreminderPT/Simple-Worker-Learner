import numpy as np
from rpc.worker import BaseWorker
class OnePolicyWorker(BaseWorker):
    def __init__(self, worker_id, network_config, policy_config, env_config, train_config):
        super().__init__(worker_id, network_config, policy_config, env_config, train_config)
    def rolling(self, num_workers):
        buffer, rewards = super().rolling(num_workers=num_workers)
        times_of_goal = len(list(filter(lambda x: x < 0, rewards)))
        game_length = len(rewards) / max(times_of_goal,1)
        sum_reward = np.sum(rewards)
        return buffer, {'rewards':sum_reward, 'game_length':game_length, 'failure times':times_of_goal}