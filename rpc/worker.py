'''
Collect buffer and transport to Learner.
'''
import pickle
import grpc
import numpy as np
from manager.manager import Manager
from rpc import policy_buffer_pb2_grpc
from rpc import policy_buffer_pb2
class WorkerClient():
    def __init__(self, Worker, worker_id, network_config, policy_config, env_config, train_config):
        self.worker = Worker(worker_id, network_config, policy_config, env_config, train_config)
    def get_policy(self, stub):
        response = stub.GetPolicy(policy_buffer_pb2.RollingResult())
        self.worker.update_policy(policy_id=0, state_dict=pickle.loads(response.state_dict))
    def rolling_and_update(self, stub):
        while True:
            buffer, reward = self.worker.rolling(num_workers=1)
            reward = np.sum(reward)
            buffer = pickle.dumps(buffer)
            response = stub.RollingAndUpdate(
                policy_buffer_pb2.RollingResult(
                    worker_id=self.worker.worker_id,
                    buffer=buffer,
                    reward=reward,
                )
            )
            self.global_step = response.global_step
            self.worker.update_policy(policy_id=0, state_dict=pickle.loads(response.state_dict))
    def run(self):
        self.worker.make()
        MAX_MESSAGE_LENGTH = 256 * 1024 * 1024
        with grpc.insecure_channel(
                'localhost:50055',
                options=[
                    ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                ]
        ) as channel:
            stub = policy_buffer_pb2_grpc.ServerStub(channel)
            self.get_policy(stub)
            self.rolling_and_update(stub)
class BaseWorker():
    def __init__(self, worker_id, network_config, policy_config, env_config, train_config):
        self.seed = network_config['seed'] + worker_id
        np.random.seed(self.seed)
        self.Buffer = train_config['Buffer']
        Policy = train_config['Policy']
        self.worker_id = worker_id
        self.network_config = network_config
        self.policy_config = policy_config
        self.device = self.network_config['device']
        self.env_config = env_config
        self.global_step = 0
        self.update_competitor = -1
        self.algorithm = self.network_config['algorithm']
        self.policy_list = [
            Policy(network_config=network_config, policy_config=policy_config, seed = self.seed).to(self.device),
            Policy(network_config=network_config, policy_config=policy_config, seed = self.seed).to(self.device)
        ]
        self.manager:Manager = None
    def update_policy(self, policy_id, state_dict):
        self.policy_list[policy_id].load_state_dict(state_dict=state_dict)
    def make(self):
        self.env_config['worker_id'] = self.worker_id
        self.manager = Manager(
            env_config=self.env_config,
            network_config=self.network_config,
            policy_configs=self.policy_config,
            seed = self.seed
        )
        self.manager.make()
    def rolling(self, num_workers):
        buffer = self.Buffer(
            network_config=self.network_config,
            policy_config=self.policy_config,
            policy_list=self.policy_list,
            num_workers=num_workers,
            seed = self.seed 
        )
        rewards = self.manager.run(buffer=buffer)
        return buffer, rewards