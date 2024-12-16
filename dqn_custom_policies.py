from typing import Any, Dict, Optional, Type

import copy

from gymnasium import spaces

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork

import torch
from torch import nn


class CustomQNetwork(QNetwork):
    """Custom action-value network for DQN. The actual network, a PyTorch Module, has to be passed as argument (q_net) to the class constructor. For documentation, refer to QNetwork.
    """

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        q_net: nn.Module,
        normalize_images: bool = True,
    ) -> None:
        super(QNetwork, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        # TODO: remove
        # self.net_arch = None # NOTE: used by the superclass methods
        # self.activation_fn = None # NOTE: used by the superclass methods

        self.features_dim = features_dim
        self.q_net = q_net

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                # net_arch=self.net_arch, # TODO: remove
                features_dim=self.features_dim,
                # activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

class CustomDQNPolicy(DQNPolicy):
    """Policy class with custom Q-value and target networks for DQN. The Q-value and target networks are obtained as separate copies of the PyTorch Module passed as argument (q_net) to the class constructor. For documentation, refer to DQNPolicy.
    """

    q_net: QNetwork
    q_net_target: QNetwork

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        q_net: nn.Module,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super(DQNPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        # TODO: remove
        # self.net_arch = None # NOTE: used by the superclass methods
        # self.activation_fn = None # NOTE: used by the superclass methods

        self.net_args = {
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            # 'net_arch': self.net_arch, # TODO: remove
            # 'activation_fn': self.activation_fn,
            'q_net': q_net,
            'normalize_images': normalize_images,
        }

        self._build(lr_schedule)
    
    def make_q_net(self) -> QNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        
        # make a copy from the original model
        # NOTE: independent Q-value and target networks required
        net_args['q_net'] = copy.deepcopy(net_args['q_net'])

        return CustomQNetwork(**net_args).to(self.device)
    
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super(DQNPolicy, self)._get_constructor_parameters()

        data.update(
            dict(
                # net_arch=self.net_args["net_arch"], # TODO: remove
                # activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data