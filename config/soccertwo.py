from buffer.mappo import MAPPOBuffer
from policy.mappo import MAPPOPolicy
from rpc.selfplay_worker import SelfPlayWorker
from train.ppo import PPO
class Config:
    
    Buffer = MAPPOBuffer
    Policy = MAPPOPolicy
    Algorithm = PPO
    Worker = SelfPlayWorker
    
    file_name = 'D:/CapstoneProject/UnitiyEnv/SoccerTwoKickGroup/UnityEnvironment.exe'
    no_graphics = True
    
    env_name = 'SoccerTwo'
    num_observations = 336
    num_actions = [3,3] 
    algorithm = 'mappo'
    seed = 20001024
    
    device = 'cuda:0'
    reward_type = 'group'
    num_steps = 256
    total_timesteps = 5_000_000
    update_epochs = 3
    learning_rate = 2.5e-4 
    anneal_lr = True
    gae_lambda = 0.95
    gamma = 0.99
    ent_coef = 0.005
    batch_size = 4096
    num_minibatches = 8 
    num_workers = 1
    target_kl = None
    clip_coef = 0.2
    clip_vloss = True
    max_grad_norm = 0.5
    vf_coef = 0.5
    norm_adv = True
    num_agents = 16 
    hidden_units = 512
    parallel_type = 'sync'
    self_play = True