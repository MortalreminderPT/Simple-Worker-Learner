
from rpc import server
from config.pyramidscollab import Config as config
train_config = {
    'Buffer':config.Buffer,
    'Policy':config.Policy,
    'Algorithm':config.Algorithm,
    'Worker':config.Worker,
}
network_config = {
    'seed':config.seed,
    'algorithm':config.algorithm,
    'hidden_units': config.hidden_units,
    'gae_lambda': config.gae_lambda,
    'batch_size': config.batch_size,
    'num_steps': config.num_steps,
    'gamma': config.gamma,
    'total_timesteps' : config.total_timesteps,
    'update_epochs' : config.update_epochs,
    'learning_rate' : config.learning_rate,
    'anneal_lr' : config.anneal_lr,
    'ent_coef' : config.ent_coef,
    'num_minibatches' : config.num_minibatches,
    'num_workers' : config.num_workers,
    'target_kl' : config.target_kl,
    'clip_coef' : config.clip_coef,
    'clip_vloss' : config.clip_vloss,
    'max_grad_norm' : config.max_grad_norm,
    'vf_coef' : config.vf_coef,
    'norm_adv' : config.norm_adv,
    'num_agents' : config.num_agents,
    'reward_type' : config.reward_type,
    'self_play' : config.self_play,
    'device':config.device,
    'parallel_type':config.parallel_type
}
policy_config = {
    'env_name':config.env_name,
    'policy_id': 0,
    'num_observations': config.num_observations,
    'num_actions': config.num_actions
}
env_config = {
    'file_name': config.file_name,
    'no_graphics': config.no_graphics
}
def print_hi(name):
    
    print(f'Hi, {name}')  
if __name__ == '__main__':
    server.serve(network_config, policy_config, train_config)
