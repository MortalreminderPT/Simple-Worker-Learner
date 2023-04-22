import time
import numpy as np
import torch
from torch import nn
class PPO():
    def __init__(self, policy_config, network_config):
        self.policy_config = policy_config
        self.network_config = network_config
        self.target_kl = self.network_config['target_kl'] 
        self.clip_coef = self.network_config['clip_coef'] 
        self.clip_vloss = self.network_config['clip_vloss']
        self.max_grad_norm = self.network_config['max_grad_norm'] 
        self.ent_coef = self.network_config['ent_coef'] 
        self.vf_coef = self.network_config['vf_coef'] 
        self.norm_adv = self.network_config['norm_adv'] 
        self.num_agents = self.network_config['num_agents'] 
        self.batch_size = self.network_config['batch_size'] 
        self.num_minibatches = self.network_config['num_minibatches'] 
        self.num_steps = self.network_config['num_steps'] 
        self.update_epochs = self.network_config['update_epochs'] 
        self.buffer = None
    def update_buffer(self, buffer):
        self.buffer = buffer
        self.batch_size = buffer.buffer_size
        self.num_steps = buffer.num_steps
    def train(self, policy, optimizer, global_step = 0, start_time = time.time()):
        envsperbatch = self.num_agents // self.num_minibatches
        envinds = np.arange(self.num_agents)
        flatinds = np.arange(self.batch_size).reshape(self.num_steps, self.num_agents)
        clipfracs = []
        if self.norm_adv:
            self.buffer.advantages_flatten = (self.buffer.advantages_flatten - self.buffer.advantages_flatten.mean()) / (self.buffer.advantages_flatten.std() + 1e-5)
        for epoch in range(self.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, self.num_agents, envsperbatch):
                end = start + envsperbatch
                mb_agent_ids = envinds[start:end]
                mb_inds = flatinds[:, mb_agent_ids].ravel()  
                _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                    self.buffer.obs_flatten[mb_inds],
                    self.buffer.global_obs_flatten[mb_inds],
                    self.buffer.actions_flatten.long()[mb_inds],
                )
                logratio = newlogprob - self.buffer.logprobs_flatten[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]
                mb_advantages = self.buffer.advantages_flatten[mb_inds]
                
                
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - self.buffer.returns_flatten[mb_inds]) ** 2
                    v_clipped = self.buffer.values_flatten[mb_inds] + torch.clamp(
                        newvalue - self.buffer.values_flatten[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - self.buffer.returns_flatten[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - self.buffer.returns_flatten[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
                optimizer.step()
            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break
        y_pred, y_true = self.buffer.values_flatten.cpu().numpy(), self.buffer.returns_flatten.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        train_states = {
            'SPS': global_step / (time.time() - start_time),
            'learning_rate': optimizer.param_groups[0]["lr"],
            'value_loss': v_loss,
            'policy_loss': pg_loss,
            'entropy': entropy.mean(),
            'old_approx_kl': old_approx_kl,
            'approx_kl': approx_kl,
            
            'explained_variance': explained_var
        }
        return policy, train_states