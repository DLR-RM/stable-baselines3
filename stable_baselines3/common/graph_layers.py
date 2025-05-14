from typing import Optional, Union, Dict, Any

import gymnasium as gym
import torch as th
from torch import nn
from gymnasium.spaces import Graph
from gymnasium.spaces.graph import GraphInstance

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BaseGraphNetwork(BaseFeaturesExtractor):
    """
    Base class for graph neural networks. This serves as a placeholder to be extended
    with specific graph neural network architectures like GCN, GAT, etc.

    :param observation_space: The Graph observation space of the environment
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: Graph,
        features_dim: int = 64,
    ) -> None:
        assert isinstance(observation_space, Graph), (
            "BaseGraphNetwork must be used with a gymnasium.spaces.Graph "
            f"observation space, not {observation_space}"
        )
        super().__init__(observation_space, features_dim)
        # This is a placeholder class and should be extended with actual graph neural network implementations
        # Users should implement their own graph neural networks by inheriting from this class

    def forward(self, observations: GraphInstance) -> th.Tensor:
        """
        Placeholder forward method. This should be implemented in a child class with actual 
        graph neural network architecture.
        
        :param observations: Graph observations from the environment
        :return: Tensor containing extracted features
        """
        # This is only a placeholder method - to be implemented by the user
        raise NotImplementedError(
            "This is a base class and doesn't implement a forward pass. "
            "Please use an actual Graph Neural Network implementation or create your own."
        )


class SimpleGraphNetwork(BaseGraphNetwork):
    """
    A simple graph neural network implementation that averages node features and edge features 
    to produce a fixed-size representation.

    :param observation_space: The Graph observation space of the environment
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: Graph,
        features_dim: int = 64,
    ) -> None:
        super().__init__(observation_space, features_dim)
        
        node_shape = None
        if hasattr(observation_space.node_space, "shape"):
            node_shape = observation_space.node_space.shape
        else:  # For Discrete spaces
            node_shape = (1,)
        
        # Define layers for node features processing
        self.node_encoder = nn.Sequential(
            nn.Linear(node_shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        
        # Define layers for the final embedding
        self.final_encoder = nn.Sequential(
            nn.Linear(32, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: GraphInstance) -> th.Tensor:
        """
        Simple graph neural network that processes node features by averaging them.
        
        :param observations: Graph observations from the environment
        :return: Tensor containing extracted features
        """
        # Process node features
        nodes = th.FloatTensor(observations.nodes)
        node_features = self.node_encoder(nodes)
        
        # Average node features to get a graph-level representation
        graph_embedding = th.mean(node_features, dim=0)
        
        # Final encoding
        output = self.final_encoder(graph_embedding)
        
        return output
