from copy import deepcopy
import numpy as np
from tqdm import tqdm

import torch
from torch.distributions import Binomial
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.dqn import DQN

import snntorch as snn


class AvgWeight(nn.Module):
    def __init__(self,
                 num_nts, # number of neurotransmitters
                 scale=1.
                 ):
        super().__init__()

        # initialize the synapse polarity matrix
        # for each neurotransmitter there exist two receptors (+ or -)
        # example (2 neurotransmitters):
        # [[  1, -1,  0,  0],
        #  [  0,  0,  1, -1]]
        S = torch.zeros(num_nts, 2*num_nts)
        S[:,::2] = torch.eye(num_nts)
        S[:,1::2] = -1*torch.eye(num_nts)

        # initialize learnable parameters
        params = torch.randn_like(S)/(np.sqrt(2)*num_nts)

        self.S = nn.Parameter(S, requires_grad=False)
        self.params = nn.Parameter(params)
        self.scale = scale

    def forward(self, T, R):
        # to probability distributions
        T, R = F.softmax(T, dim=-1), F.softmax(R, dim=-1)

        # compute neurotransmitter-receptor conductances
        C = self.scale*self.S*F.sigmoid(self.params)

        # compute average synapse conductances
        C_avg = torch.matmul(torch.matmul(T, C), R.T)

        return C_avg
    
class SynMultiplicity(nn.Module):
    def __init__(self,
                 num_genes,
                 temperature=1.,
                 rules=None # NOTE: has to be a boolean tensor
                 ):
        super().__init__()

        # check matching between rules and number of genes
        if rules is not None and\
            rules.shape != (num_genes, num_genes):
            raise Exception(f'The # of genes is expected to be {rules.shape[0]}')

        # initialize the matrix of genetic rules
        O = torch.randn(num_genes, num_genes)/num_genes

        # set the parameters of the sigmoids used when the bio-plausible genetic rules are provided
        # NOTE: the learnable parameters corresponding to the co-expressed genes will be mapped to [.5, 1], the others to [0, .5]. The idea is to assign high probabilities to the co-expressed pairs only. See the forward pass implementation for more details.
        if rules is not None:
            self.rules = True
            
            init_std = 1/num_genes
            x_translation = 3*init_std*torch.ones_like(O)
            x_translation[rules] *= -1

            y_translation = torch.zeros_like(O)
            y_translation[rules] = .5

            self.x_trans = nn.Parameter(x_translation, requires_grad=False)
            self.y_trans = nn.Parameter(y_translation, requires_grad=False)

        self.O = nn.Parameter(O)
        self.temperature = temperature

    def forward(self, X, Y):
        # to positive real numbers
        X = X**2
        Y = Y**2

        O = self.O

        # shift the sigmoid horizontally (for bio-plausible rules only)
        if self.rules:
            O = O - self.x_trans
        
        # to probabilities
        O = F.sigmoid(O/self.temperature)

        # shift the sigmoid vertically (for bio-plausible rules only)
        if self.rules:
            O = .5*O + self.y_trans

        # compute average synapse multiplicities
        B = torch.matmul(torch.matmul(X, O), Y.T)

        return B

class XGEM(nn.Module):
    def __init__(self, layer_sizes, num_genes, num_nts,
        O_temperature=1., C_scale=1., rules=None):
        super().__init__()
        
        # initialize expression patterns
        Xs = nn.ParameterList()
        for size in layer_sizes:
            X = torch.zeros(size, num_genes)
            nn.init.kaiming_normal_(X.T, mode='fan_out')
            Xs.append(X)

        # initialize neurotransmitter distributions
        Ts = nn.ParameterList()
        for size in layer_sizes[:-1]:
            T = torch.zeros(size, num_nts)
            nn.init.kaiming_normal_(T.T, mode='fan_out')
            Ts.append(T)

        # initialize receptor distributions
        Rs = nn.ParameterList()
        for size in layer_sizes[1:]:
            R = torch.zeros(size, 2*num_nts)
            nn.init.kaiming_normal_(R.T, mode='fan_out')
            Rs.append(R)

        # initialize biases
        biases = nn.ParameterList()
        for size_pre, size_post in zip(layer_sizes[:-1], layer_sizes[1:]):
            b = torch.zeros(size_pre + 1, size_post)
            nn.init.kaiming_normal_(b.T, mode='fan_in')
            biases.append(b[-1])

        self.Xs = Xs
        self.Ts = Ts
        self.Rs = Rs
        self.biases = biases

        # initialize modules for computing synapse conductances and multiplicities
        self.avg_weight = AvgWeight(num_nts, scale=C_scale)
        self.syn_multi = SynMultiplicity(num_genes, temperature=O_temperature, rules=rules)

    def forward(self, x):
        for i, (X_in, X_out, T, R, bias) in enumerate(zip(
            self.Xs[:-1],
            self.Xs[1:],
            self.Ts,
            self.Rs,
            self.biases
        )):
            # compute average synapse conductances and multiplicities
            C_avg = self.avg_weight(T, R)
            B = self.syn_multi(X_in, X_out)
            
            # compute the weight matrix (conductances)
            W = C_avg*B
            
            # linear layer
            x = torch.matmul(x, W) + bias

            # activation function
            if i < len(self.Xs) - 2:
                x = F.selu(x)
        
        return x

class SampledNet(nn.Module):
    def __init__(self, weights, biases):
        super().__init__()

        self.Ws = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases)

    def forward(self, x):
        for i, (W, bias) in enumerate(zip(
            self.Ws,
            self.biases
        )):    
            # linear layer
            x = torch.matmul(x, W) + bias

            # activation function
            if i < len(self.Ws) - 1:
                x = F.selu(x)
        
        return x
    
def init_expression(
        Xs, # gene expression patterns
        Qs, # neurotransmitter distributions
        Rs, # receptor distributions
        biases, # neuron biases
        seed=1
):
    """Initializes the provided gene expression patterns according to the bio-plausible procedure proposed by Kerstjens et al..
    """

    torch.manual_seed(seed)

    num_neurons = sum([X.shape[0] for X in Xs])
    num_genes = sum([tensors[0].shape[1] for tensors in [Xs, Qs, Rs]]) + 1 # NOTE: the +1 is for biases

    # initialize the zygote's gene expression pattern
    cs = torch.randn(1, num_genes)

    # assign expression patterns according to a simulated lineage tree
    while cs.shape[0] < num_neurons:
        cs_new = torch.zeros(0, num_genes)
        for c in cs:
            delta_1 = torch.randn(1, num_genes)
            delta_2 = torch.randn(1, num_genes)

            cs_new = torch.cat([
                cs_new,
                c + delta_1,
                c + delta_2
            ], dim=0)
        
        cs = cs_new

    cs = cs[:num_neurons]

    # shuffle patterns preserving "spatial continuity"
    cs = torch.roll(cs, shifts=np.random.randint(0, num_neurons), dims=0)

    # rescale biases to restore the original distribution
    biases_vec = torch.cat(list(biases)).detach().cpu()
    target_b_mean, target_b_std = torch.mean(biases_vec), torch.std(biases_vec)
    b_mean, b_std = torch.mean(cs[:,-1]), torch.std(cs[:,-1])
    cs[:,-1] = target_b_mean + target_b_std*(cs[:,-1] - b_mean)/b_std

    # separate expression patterns by layer
    layers = []
    for X in Xs:
        layer_size = X.shape[0]
        layers.append(cs[:layer_size])
        cs = cs[layer_size:]

    def set_expression(tensors, layers, idx_start, idx_end):
        """Assigns the correct expression patterns to the given parameter tensor and returns the updated expression patterns from which the used ones have been removed.
        """
        
        num_genes_tensor = tensors[0].shape[1]
        layers_tensor = [l[:,:num_genes_tensor] for l in layers[idx_start:idx_end]]
        layers = [l[:,num_genes_tensor:] for l in layers]
        for tensor, l in zip(tensors, layers_tensor):
            tensor.data = l.to(tensor.device)

        return layers
    
    # assign the initialized patterns to the pattern tensors provided
    num_layers = len(Xs)
    layers = set_expression(Xs, layers, idx_start=0, idx_end=num_layers)
    layers = set_expression(Qs, layers, idx_start=0, idx_end=num_layers - 1)
    layers = set_expression(Rs, layers, idx_start=1, idx_end=num_layers)

    # assign the initialized biases to the bias tensors provided
    for b, l in zip(biases, layers[1:]):
        b.data = l.squeeze().to(b.device)

def simulate_and_evaluate(
    model: DQN,
    env,
    n_samplings=30,
    n_eval_episodes=10,
    avg_degree=1e4,
    return_best_model=False,
    pbar=False
):
    """Simulates synaptogenesis by sampling synapses from the distributions encoded in the provided model and evaluate the networks obtained on the given environment. The function returns a list of mean rewards obtained through the evaluations, the total number of steps performed during the evaluations and, optionally, the best sampled network.
    NOTE: In this implementation there is no sampling for neurotransmitters and receivers.
    """

    q_net = model.q_net.q_net.cpu()
    O = q_net.syn_multi.O

    # shift the sigmoid horizontally (for bio-plausible rules only)
    try: O = O - q_net.syn_multi.x_trans
    except: pass
    
    # to probabilities
    O = F.sigmoid(O/q_net.syn_multi.temperature)

    # shift the sigmoid vertically (for bio-plausible rules only)
    try: O = .5*O + q_net.syn_multi.y_trans
    except: pass

    # compute the multiplicative corrective factor for the binomials' "number of experiments" parameter
    # the correction aims to obtain, by rescaling the network's adjacency matrix, the provided average degree
    # NOTE: if before rounding n < 0.5, rounding will fail sampling
    # NOTE: higher synaptic counts and lower synaptic weights reduce the sampling-induced quantization error
    num_neurons = 0
    num_synapses = 0
    for X_in, X_out in zip(
        q_net.Xs[:-1],
        q_net.Xs[1:],
    ):
        num_neurons += X_out.shape[0]
        num_synapses += q_net.syn_multi(X_in, X_out).sum()
    num_neurons += q_net.Xs[0].shape[0]
    correction = (avg_degree*num_neurons)/(2*num_synapses)

    rewards = []
    max_reward = float('-inf')
    steps_performed = 0
    for _ in tqdm(range(n_samplings)) if pbar else range(n_samplings):
        Ws = []
        for X_in, X_out, T, R in zip(
            q_net.Xs[:-1],
            q_net.Xs[1:],
            q_net.Ts,
            q_net.Rs
        ):
            # compute average synapse conductances
            C_avg = q_net.avg_weight(T, R)

            # sample the number of synapses between each neuron pair and for each gene pair
            B = torch.zeros(X_in.shape[0], X_out.shape[0])
            for i in range(X_in.shape[1]):
                for j in range(X_out.shape[1]):
                    n = torch.outer(X_in[:,i]**2, X_out[:,j]**2)
                    n = torch.round(correction*n)

                    bin = Binomial(n, O[i,j])
                    B += bin.sample()
            
            # compute the weight matrix (conductances)
            W = (1/correction)*C_avg*B

            Ws.append(W)
        
        # initialize the sampled network
        sampled_q_net = SampledNet(Ws, q_net.biases).to(model.device)
        
        # evaluate
        model.q_net.q_net = sampled_q_net
        episode_rewards, episode_lengths = evaluate_policy(
            model,
            env,
            n_eval_episodes=n_eval_episodes,
            return_episode_rewards=True
        )
        mean_reward = np.mean(episode_rewards)
        rewards.append(mean_reward)

        if mean_reward > max_reward:
            best_model = deepcopy(model)
            max_reward = mean_reward

        steps_performed += np.sum(episode_lengths)

    if return_best_model:
        return rewards, steps_performed, best_model
    else:
        return rewards, steps_performed