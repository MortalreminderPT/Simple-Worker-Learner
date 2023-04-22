'''
Update policy params to workers.
'''
import pickle
import queue
import threading
import time
from concurrent import futures
import grpc
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from rpc import policy_buffer_pb2_grpc, policy_buffer_pb2
def write_state(writer, train_stats, reward_dict, global_step):
    writer.add_scalar("charts/learning_rate", train_stats['learning_rate'], global_step)
    writer.add_scalar("losses/value_loss", train_stats['value_loss'], global_step)
    writer.add_scalar("losses/policy_loss", train_stats['policy_loss'], global_step)
    writer.add_scalar("losses/entropy", train_stats['entropy'], global_step)
    writer.add_scalar("losses/old_approx_kl", train_stats['old_approx_kl'], global_step)
    writer.add_scalar("losses/approx_kl", train_stats['approx_kl'], global_step)
    
    writer.add_scalar("losses/explained_variance", train_stats['explained_variance'], global_step)
    writer.add_scalar("charts/SPS", train_stats['SPS'], global_step)
    for reward_key, reward_item in reward_dict.items():
        writer.add_scalar(f"reward/{reward_key}", reward_item, global_step)
class ServerDB:
    def __init__(self, network_config, policy_config, train_config):
        Policy = train_config['Policy']
        Algorithm = train_config['Algorithm']
        self.network_config = network_config
        self.policy_config = policy_config
        self.seed = network_config['seed']
        self.device = self.network_config['device']
        torch.manual_seed(self.seed)
        self.total_timesteps = self.network_config['total_timesteps']
        self.num_steps = self.network_config['num_steps']
        self.learning_rate = self.network_config['learning_rate']
        self.num_workers = self.network_config['num_workers']
        self.anneal_lr = self.network_config['anneal_lr']
        self.parallel_type = self.network_config['parallel_type']
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.event.set()
        self.algorithm = self.network_config['algorithm']
        self.policy = Policy(network_config=network_config, policy_config=policy_config, seed=self.seed).to(self.device)
        state_dict =pickle.loads(torch.load('./model/PyramidsCollab.pt', ))
        self.policy.load_state_dict(state_dict)
        self.ppo_lstm = Algorithm(policy_config, network_config)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001, eps=1e-5)
        self.global_step = 0
        self.update = 1
        self.num_updates = self.total_timesteps // self.num_steps
        self.train_queue = queue.Queue(maxsize=self.num_workers)
        self.state_dict = pickle.dumps(self.policy.state_dict())
        env_id = "PushBlock_N"
        self.env_name = self.policy_config['env_name']
        exp_name = "PT"
        seed = 0
        run_name = f"{env_id}__{exp_name}__{self.algorithm}__{seed}__{int(time.time())}"
        self.writer = SummaryWriter(f"runs/{run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars().items()])),
        )
class ServerServicer(policy_buffer_pb2_grpc.ServerServicer):
    def __init__(self, server_db):
        self.db = server_db
    def GetPolicy(self, request, context):
        if self.db.parallel_type == 'sync':
            self.db.event.wait()
        return policy_buffer_pb2.Policy(
            state_dict=self.db.state_dict
        )
    def RollingAndUpdate(self, request, context):
        buffer = pickle.loads(request.buffer)
        if self.db.global_step > self.db.total_timesteps:
            print(f'训练已完成，步数{self.db.global_step}')
            return policy_buffer_pb2.Policy(
                global_step=self.db.global_step,
                state_dict=self.db.state_dict
            )
        with self.db.lock:
            self.db.event.clear()
            self._train(id=request.id, buffer=buffer, reward_dict=request.reward)
        if self.db.parallel_type == 'sync':
            self.db.event.wait()
        return policy_buffer_pb2.Policy(
            global_step=self.db.global_step,
            state_dict=self.db.state_dict
        )
    def _train(self, id, buffer, reward_dict):
        if self.db.anneal_lr:
            frac = 1.0 - (self.db.update - 1.0) / self.db.num_updates
            lrnow = frac * self.db.learning_rate
            self.db.optimizer.param_groups[0]["lr"] = lrnow
        self.db.ppo_lstm.update_buffer(buffer=buffer)
        self.db.global_step += self.db.num_steps
        self.db.policy, train_state = self.db.ppo_lstm.train(
            policy=self.db.policy,
            optimizer=self.db.optimizer,
            global_step=self.db.global_step
        )
        write_state(writer=self.db.writer, train_stats=train_state, reward_dict=reward_dict,
                    global_step=self.db.global_step)
        print(f'开始训练，第{self.db.global_step}轮')
        self.db.update += 1
        self.db.train_queue.put(id)
        if self.db.train_queue.full():
            self.db.train_queue.queue.clear()
            self.db.state_dict = pickle.dumps(self.db.policy.state_dict())
            torch.save(self.db.state_dict, f'./model/{self.db.env_name}.pt')
            self.db.event.set()
def serve(network_config, policy_config, train_config):
    MAX_MESSAGE_LENGTH = 256 * 1024 * 1024
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ]
    )
    server_db = ServerDB(network_config, policy_config, train_config)
    policy_buffer_pb2_grpc.add_ServerServicer_to_server(ServerServicer(server_db), server)
    server.add_insecure_port('[::]:50055')
    server.start()
    print('server start')
    server.wait_for_termination()
if __name__ == '__main__':
    serve()