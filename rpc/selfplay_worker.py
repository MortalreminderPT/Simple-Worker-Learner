import queue
import numpy as np
from rpc.worker import BaseWorker
class SelfPlayWorker(BaseWorker):
    def __init__(self, worker_id, network_config, policy_config, env_config, train_config):
        super().__init__(worker_id, network_config, policy_config, env_config, train_config)
        self.self_play = self.network_config['self_play']
        self.save_steps = 50000 
        self._save_steps = 0
        self.swap_steps = 2560 
        self._swap_steps = 0
        self.play_against_latest_model_ratio = 0.5 
        self.window = 10
        self.policy_snaps = queue.Queue(maxsize=self.window)
    def match(self):
        if self.self_play:
            if self._save_steps < self.global_step:
                self._save_steps += self.save_steps
                if self.policy_snaps.full():
                    self.policy_snaps.get()
                self.policy_snaps.put(self.policy_list[0].state_dict())
                print('保存策略')
            if self._swap_steps < self.global_step:
                self._swap_steps += self.swap_steps
                self.update_policy(policy_id=1, state_dict=self._find_policy(self.play_against_latest_model_ratio))
                print('切换对手')
        return
    def _find_policy(self, training_policy_weights):
        weights = [(1-training_policy_weights) / self.policy_snaps.qsize()] * self.policy_snaps.qsize()
        weights[-1] = training_policy_weights
        import random
        return random.choices(self.policy_snaps.queue, weights=weights, k=1)[0]
    def rolling(self, num_workers):
        self.match()
        buffer, rewards = super().rolling(num_workers=num_workers)
        times_of_goal = len(list(filter(lambda x: x < 0, rewards)))
        game_length = len(rewards) / max(times_of_goal,1) 
        sum_reward = np.sum(rewards)
        return buffer, {'rewards':sum_reward, 'game_length':game_length, 'failure times':times_of_goal}