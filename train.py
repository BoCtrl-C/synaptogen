from dqn_custom_policies import CustomDQNPolicy
from hyperparams import HYPERPARAMETERS
from models import XGEM
from utils import VideoRecorderCallback

import argparse
import os

import numpy as np

import torch
import torch.nn as nn

import gym

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    EvalCallback,
    EveryNTimesteps
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import wandb
from wandb.integration.sb3 import WandbCallback

from multiprocessing import Process


PROJECT = 'synaptogen'
CONFIG= {
    'method': 'grid',
    'metric': {
        'name': 'eval/mean_reward',
        'goal': 'maximize'
    },
    'parameters': {
        'num_genes': {'values': [16, 32, 64]},
        'num_nts': {'values': [3]},
        'learning_rate': {'values': [3e-2, 3e-3, 3e-4]},
        'seed': {'values': [1, 2, 3]}
    }
}
HIDDEN_SIZE = 128
M = 5 # multiplicative factor for the default number of training steps
NUM_PROCESSES = 3


parser = argparse.ArgumentParser()
parser.add_argument('-q', '--qnet', type=str)
parser.add_argument('-e', '--env', type=int)
parser.add_argument('-c', '--cuda', type=int)
parser.add_argument('--bio', action='store_true')
parser.add_argument('--snn', action='store_true')
args = parser.parse_args()

# use the right model if SNNs are requested
if args.snn: from models import SpikingXGEM as XGEM

ENV_NAME = list(HYPERPARAMETERS.keys())[args.env]

# load the biological genetic rules
if args.bio:
    npz = np.load('data/genetic_rules.npz', allow_pickle=True)
    O = torch.tensor(npz['O']).bool()

    # set the correct number of genes
    CONFIG['parameters']['num_genes']['values'] = [O.shape[0]]

wandb.login()

sweep_id = wandb.sweep(CONFIG, project=PROJECT)

def make_env():
    env = gym.make(ENV_NAME)
    env = Monitor(env)
    return env
env = DummyVecEnv([make_env])
features_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

hyperparameters = HYPERPARAMETERS[ENV_NAME]

def train():
    CKPTS_DIR = 'ckpts'
    if args.bio: CKPTS_DIR += '-bio'
    if args.snn: CKPTS_DIR += '-snn'

    run = wandb.init(sync_tensorboard=True)
    config = wandb.config

    num_genes = config['num_genes']
    num_nts = config['num_nts']
    seed = config['seed']

    hyperparameters['learning_rate'] = config['learning_rate']

    # TODO: restore
    # # restore the original eps scheduler
    # hyperparameters['exploration_fraction'] = hyperparameters['exploration_fraction']/M

    torch.manual_seed(seed) # weights initialization seed
    if args.qnet == 'mlp':
        q_net = nn.Sequential(
            nn.Linear(features_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, action_dim)
        )
    elif args.qnet == 'gem':
        q_net = XGEM(
            layer_sizes=[features_dim, HIDDEN_SIZE, action_dim],
            num_genes=num_genes,
            num_nts=num_nts,
            O_temperature=.1,
            C_scale=1.,
            rules=O if args.bio else None # inject the genetic rules
        )
    else:
        raise Exception('Invalid Q network')

    model = DQN(
        CustomDQNPolicy,
        env,
        verbose=1,
        tensorboard_log=f'runs/{run.id}',
        policy_kwargs={'q_net': q_net},
        device=f'cuda:{args.cuda}',
        seed=seed,
        **hyperparameters
    )
    
    model.learn(
        total_timesteps=M*1e5,
        callback=[
            WandbCallback(verbose=2),
            EveryNTimesteps(
                n_steps=1e4,
                callback=VideoRecorderCallback(ENV_NAME)
            ),
            EvalCallback(
                eval_env=model.env,
                eval_freq=1e4,
                best_model_save_path=os.path.join(CKPTS_DIR, run.id)
            )
        ]
    )

    run.finish()

# parallelize sweep on multiple processes
def run_agent():
    wandb.agent(sweep_id, function=train, project=PROJECT)

# create a list to store the processes
processes = []

# start the parallel processes
for _ in range(NUM_PROCESSES):
    process = Process(target=run_agent)
    process.start()
    processes.append(process)

# wait for all processes to finish
for process in processes:
    process.join()