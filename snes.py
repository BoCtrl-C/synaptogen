# FIXME: when cuda:0 is selected, the script may run on the cpu


import sys
sys.path.append('pybrain')

from dqn_custom_policies import CustomDQNPolicy
from hyperparams import HYPERPARAMETERS
from models import XGEM, simulate_and_evaluate
from utils import s_to_h_min_s

import argparse
from functools import partial
import gym
from multiprocessing import Process
import numpy as np
import os
import pandas as pd
from pybrain.optimization import SNES
from stable_baselines3 import DQN
import time
import torch
import torch.nn as nn
import wandb
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


HIDDEN_SIZE = 128
N_EVAL_EPISODES = 10
MAX_STEPS = 5e5
AVG_DEGREE = 1e5
NUM_PROCESSES = 3
CKPT_DIR = 'ckpts-snes'


def get_param_list(model: DQN):
    """Returns a list with all the nn.Parameter tensors of the given model.
    """
    
    param_list = []
    for model_param in [
        model.q_net.q_net.Xs,
        model.q_net.q_net.Ts,
        model.q_net.q_net.Rs,
        model.q_net.q_net.biases,
        model.q_net.q_net.syn_multi.O,
        model.q_net.q_net.avg_weight.params
    ]:

        if type(model_param) == nn.ParameterList:
            for element in model_param:
                if element.requires_grad: param_list.append(element)
        elif type(model_param) == nn.Parameter:
            if model_param.requires_grad: param_list.append(model_param)
        else:
            raise Exception('Invalid parameter type')
        
    return param_list

def vectorize(param_list):
    """Flattens and concatenates the given parameters.
    """
    
    x = torch.zeros(0)
    for param in param_list:
        x = torch.cat([x, param.view(-1).cpu()], dim=0)

    return x.detach()

def unvectorize(x, param_list):
    """Updates the parameters of the model from which the parameter list has been extracted using the values in x.
    """

    for param in param_list:
        x_param, x = x[:param.numel()], x[param.numel():]
        param.data = x_param.view_as(param)

class CustomSNES(SNES):
    """SNES optimizer customized for stopping condition based on the number of steps performed into the agent's environment.
    """

    num_steps = 0
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.maxEvaluations = float('inf')
        self.maxSteps = 1e6

    def _stoppingCriterion(self):
        return CustomSNES.num_steps >= self.maxSteps\
            or super()._stoppingCriterion()

    def _notify(self):
        # W&B logging
        wandb.log({
            'fitness': self.bestEvaluation,
            'steps': CustomSNES.num_steps
        })

        if self.verbose:
            print((
                'Epoch:', self.numLearningSteps,
                'elapsed time:', s_to_h_min_s(time.time() - self.start_time),
                'steps:', CustomSNES.num_steps,
                'best:', self.bestEvaluation
            ))
        if self.listener is not None:
            self.listener(self.bestEvaluable, self.bestEvaluation)

    def learn(self, *args, **kwargs):
        # reset the step counter and start the timer
        CustomSNES.num_steps = 0
        self.start_time = time.time()

        return super().learn(*args, **kwargs)

def objF(x, env_name, layer_sizes, num_genes, num_nts, n_samplings, rules=None, device='cpu'):
    """Function to maximize through SNES.
    NOTE: the function has been designed to work with parallel calls.
    """
    
    # initialize the environment
    env = gym.make(env_name)

    # initialize the agent
    q_net = XGEM(
        layer_sizes=layer_sizes,
        num_genes=num_genes,
        num_nts=num_nts,
        O_temperature=.1,
        C_scale=1.,
        rules=rules
    )
    model = DQN(
        CustomDQNPolicy,
        env,
        policy_kwargs={'q_net': q_net},
        device=device
    )
    
    # update the model parameters
    unvectorize(
        torch.tensor(x, device=device).float(),
        get_param_list(model)
    )

    # simulate synaptogenesis by sampling from the learned distributions
    rewards, steps_performed = simulate_and_evaluate(
        model,
        env,
        n_samplings=n_samplings,
        n_eval_episodes=N_EVAL_EPISODES,
        avg_degree=AVG_DEGREE
    )
    CustomSNES.num_steps += steps_performed
    return np.mean(rewards)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', type=int)
    parser.add_argument('--init', action='store_true')
    parser.add_argument('-c', '--cuda', type=int)
    parser.add_argument('--bio', action='store_true')
    args = parser.parse_args()

    if args.bio: CKPT_DIR += '-bio'

    # create the ckpt directory if it does not exist
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)

    PROJECT = 'synaptogen'
    CONFIG= {
        'method': 'grid',
        'metric': {
            'name': 'test/mean_reward',
            'goal': 'maximize'
        },
        'parameters': {
            'env_name': {'values': [list(HYPERPARAMETERS.keys())[args.env]]},
            'num_genes': {'values': [16, 32, 64]},
            'num_nts': {'values': [3]},
            'batch_size': {'values': [8, 16, None]},
            'n_samplings': {'values': [10, 20, 30]}
        }
    }
    DEVICE = f'cuda:{args.cuda}'

    # load the biological genetic rules
    if args.bio:
        npz = np.load('data/genetic_rules.npz', allow_pickle=True)
        O = torch.tensor(npz['O']).bool()

        # set the correct number of genes
        CONFIG['parameters']['num_genes']['values'] = [O.shape[0]]

    wandb.login()
    sweep_id = wandb.sweep(CONFIG, project=PROJECT)

    def optimize():
        run = wandb.init()
        config = wandb.config

        env_name = config['env_name']
        num_genes = config['num_genes']
        num_nts = config['num_nts']
        batch_size = config['batch_size']
        n_samplings = config['n_samplings']

        # initialize the environment in which the agent will act
        env = gym.make(env_name)
        features_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # initialize the agent
        if args.init:
            # load the best agent trained with gradient descent
            SHOWN_COLS = ['env_name', 'run_id', 'num_genes', 'learning_rate', 'seed', 'mean_reward', 'reward_std']
            df = pd.read_csv('runs.csv')
            best_run = df[(df['env_name']==env_name)&(df['num_genes']==num_genes)][SHOWN_COLS].sort_values(
                by=['mean_reward', 'reward_std', 'num_genes'],
                ascending=[False, True, True]
            ).head(1)
            run_id = best_run['run_id'].item()

            model = DQN.load(os.path.join('ckpts', run_id, 'best_model.zip'), device=DEVICE)
        else:
            q_net = XGEM(
                layer_sizes=[features_dim, HIDDEN_SIZE, action_dim],
                num_genes=num_genes,
                num_nts=num_nts,
                O_temperature=.1,
                C_scale=1.,
                rules=O if args.bio else None # inject the genetic rules
            )
            model = DQN(
                CustomDQNPolicy,
                env,
                policy_kwargs={'q_net': q_net},
                device=DEVICE
            )

        # set the initial guess
        x0 = vectorize(get_param_list(model))
        print(f'The opt. space is {x0.numel()}-dim')

        # initialize the optimizer
        l = CustomSNES(partial(
            objF,
            env_name=env_name,
            layer_sizes=[features_dim, HIDDEN_SIZE, action_dim],
            num_genes=num_genes,
            num_nts=num_nts,
            n_samplings=n_samplings,
            rules=O if args.bio else None,
            device=DEVICE
        ), x0.numpy())
        l.minimize = False
        l.maxSteps = MAX_STEPS
        if batch_size: l.batchSize = batch_size
        l.verbose = True

        # optimize
        x, reward = l.learn()

        # test the found optimal agent
        unvectorize(
            torch.tensor(x, device=DEVICE).float(),
            get_param_list(model)
        )
        rewards, _ = simulate_and_evaluate(model, env, avg_degree=AVG_DEGREE)
        wandb.log({
            'test/mean_reward': np.mean(rewards),
            'test/reward_std': np.std(rewards),
            'test/max_reward': np.max(rewards)
        })

        # save the optimal parameters
        np.save(os.path.join(CKPT_DIR, f'x-{run.id}.npy'), x)
        print(f'Best mean reward achieved: {reward}')

        run.finish()

    # parallelize sweep on multiple processes
    def run_agent():
        wandb.agent(sweep_id, function=optimize, project=PROJECT)

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