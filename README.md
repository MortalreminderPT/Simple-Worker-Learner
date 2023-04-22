# Simple-Worker-Learner

Simple-Worker-Learner is a lightweight framework for multi-agent reinforcement learning based on the Worker-Learner architecture. This framework includes a distributed implementation of the MAPPO algorithm, which stands for Multi-Agent Proximal Policy Optimization. It supports Unity ML-Agents environment as a training environment.

## Introduction

The MAPPO algorithm is a state-of-the-art algorithm for multi-agent reinforcement learning, which extends the Proximal Policy Optimization (PPO) algorithm to the multi-agent setting. It uses a centralized critic to estimate the value of the joint action space, and a decentralized actor to learn individual policies that optimize the joint objective. MAPPO has shown impressive performance on a variety of multi-agent scenarios, such as cooperative navigation, competitive games, and communication tasks.

Unity ML-Agents is a plugin for Unity game engine that allows developers to create intelligent agents that can learn from and interact with their environment. It provides a flexible and powerful framework for designing and training game AI, including support for various sensors, actions, and rewards. With Unity ML-Agents, you can create realistic and challenging game scenarios that require intelligent behavior and decision-making.

To learn more about the MAPPO algorithm and Unity ML-Agents, please refer to the following resources:

- MAPPO: [Multi-Agent Proximal Policy Optimization](https://arxiv.org/abs/1910.01741)
- Unity ML-Agents: [Unity Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents)

## Usage

To use Simple-Worker-Learner, you need to follow these steps:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/MortalreminderPT/Simple-Worker-Learner.git
```

2. Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

3. Prepare your Unity ML-Agents environment and configure the training parameters in the `config.yml` file.

4. Run the main script on the master node:

```bash
python main.py
```

5. Run the worker script on the slave nodes:

```bash
python workers.py
```

6. Monitor the training progress in the console or using TensorBoard.

You can simply modify the configuration files in the `config` folder.  You can also define your own agent behaviors and reward functions by subclassing the `Agent` class.

If you encounter any issues or have any suggestions, please feel free to open an issue or a pull request on GitHub.

## License

Simple-Worker-Learner is released under the GPLv2 License. See the `LICENSE` file for more details.
