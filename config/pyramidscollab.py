from buffer.mappo import MAPPOBuffer
from policy.mappo import MAPPOPolicy
from rpc.onepolicy_worker import OnePolicyWorker
from train.ppo import PPO
class Config:
    
    Buffer = MAPPOBuffer
    Policy = MAPPOPolicy
    Algorithm = PPO
    Worker = OnePolicyWorker
    
    file_name = 'D:/CapstoneProject/UnitiyEnv/PyramidsCollabTest/UnityEnvironment.exe'
    no_graphics = False
    
    env_name = 'PyramidsCollab'
    num_observations = 193
    num_actions = [1, 4] 
    algorithm = 'mappo'
    seed = 20001024
    
    device = 'cuda:0'
    reward_type = 'group'
    num_steps = 512
    total_timesteps = 5_000_000
    update_epochs = 3
    learning_rate = 2.5e-4 
    anneal_lr = True
    gae_lambda = 0.95
    gamma = 0.99
    ent_coef = 0.005
    batch_size = 16384
    num_minibatches = 32 
    num_workers = 1
    target_kl = None
    clip_coef = 0.2
    clip_vloss = True
    max_grad_norm = 0.5
    vf_coef = 0.5
    norm_adv = True
    num_agents = 32 
    hidden_units = 256
    parallel_type = 'sync'
    self_play = True