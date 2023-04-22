import numpy as np
import torch
class Buffer(object):
    def __init__(self, network_config, policy_config, seed = 0):
        self.network_config = network_config
        self.policy_config = policy_config
        self.seed = seed
        self.device = self.network_config['device']
        torch.manual_seed(seed)
        np.random.seed(seed)
        self._step = 0
    def update_actions(self, **kwargs): pass
    def update_rewards(self, **kwargs): pass
    def calculate(self, **kwargs): pass
    def flatten(self, **kwargs): pass
