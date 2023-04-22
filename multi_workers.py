from threading import Thread
from main import network_config, policy_config, env_config, train_config
from rpc.worker import WorkerClient
class W():
    def start_worker(worker_id):
        worker = WorkerClient(Worker=train_config['Worker'], worker_id=worker_id, network_config=network_config, policy_config=policy_config, env_config=env_config, train_config=train_config)
        worker.run()
if __name__ == '__main__':
    threads = []
    for i in range(network_config['num_workers']):
        a = Thread(target=W.start_worker, args=[i])
        a.start()
        threads.append(a)
