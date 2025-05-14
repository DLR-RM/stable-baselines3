import warnings

import numpy as np
import pytest
import gymnasium as gym
import gymnasium.spaces as sp
from gymnasium.utils.env_checker import check_env
import torch as th
from gymnasium.spaces.graph import GraphInstance

from stable_baselines3.common.env_checker import check_env as sb3_check_env
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.graph_layers import SimpleGraphNetwork
from stable_baselines3.ppo import PPO


class GraphEnv(gym.Env):
    """
    Custom gym environment with Graph observation space for testing.
    
    :param include_box_obs: Whether to include a Box observation space alongside the Graph
    """
    def __init__(self, include_box_obs=True):
        super().__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Define node and edge shapes
        node_shape = (2,)
        edge_shape = (3,)
        
        # Create observation space including a graph
        if include_box_obs:
            self.observation_space = sp.Dict({
                "my_graph": sp.Graph(
                    node_space=sp.Box(low=0, high=1, shape=node_shape),
                    edge_space=sp.Box(low=0, high=1, shape=edge_shape)
                ),
                "my_box": sp.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            })
        else:
            self.observation_space = sp.Dict({
                "my_graph": sp.Graph(
                    node_space=sp.Box(low=0, high=1, shape=node_shape),
                    edge_space=sp.Box(low=0, high=1, shape=edge_shape)
                ),
            })

        # Create a constant observation for testing
        self.const_obs = {
            "my_graph": GraphInstance(
                nodes=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
                edges=np.array([[0.5, 0.6, 0.7]], dtype=np.float32),
                edge_links=np.array([[0, 1]], dtype=np.int64)
            ),
        }

        if include_box_obs:
            self.const_obs["my_box"] = np.array([0.2], dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        # Always return a copy to avoid accidental mutation
        obs = {}
        for k, v in self.const_obs.items():
            if isinstance(v, np.ndarray):
                obs[k] = v.copy()
            elif isinstance(v, GraphInstance):
                # Ensure all GraphInstance fields are arrays of correct dtype
                obs[k] = GraphInstance(
                    nodes=np.array(v.nodes, dtype=np.float32),
                    edges=np.array(v.edges, dtype=np.float32),
                    edge_links=np.array(v.edge_links, dtype=np.int64)
                )
            else:
                obs[k] = v
        info = {}
        return obs, info

    def step(self, action):
        obs = {}
        for k, v in self.const_obs.items():
            if isinstance(v, np.ndarray):
                obs[k] = v.copy()
            elif isinstance(v, GraphInstance):
                obs[k] = GraphInstance(
                    nodes=np.array(v.nodes, dtype=np.float32),
                    edges=np.array(v.edges, dtype=np.float32),
                    edge_links=np.array(v.edge_links, dtype=np.int64)
                )
            else:
                obs[k] = v
        reward = 1.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


class MyExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for handling Graph observation spaces.
    Uses SimpleGraphNetwork for graph observations and a simple MLP for box observations.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=10):
        super().__init__(observation_space, features_dim=1)
        extractors = {}
        for key, subspace in observation_space.spaces.items():
            if isinstance(subspace, sp.Graph):
                extractors[key] = SimpleGraphNetwork(subspace, features_dim=features_dim)
            elif isinstance(subspace, sp.Box):
                extractors[key] = th.nn.Sequential(
                    th.nn.Linear(int(np.prod(subspace.shape)), features_dim),
                    th.nn.ReLU()
                )
        self.extractors = th.nn.ModuleDict(extractors)
        self._features_dim = features_dim * len(extractors)

    def forward(self, observations):
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if isinstance(self.extractors[key], th.nn.Sequential):
                # Only convert Box to tensor
                if isinstance(obs, np.ndarray):
                    obs = th.from_numpy(obs).float()
                # Always ensure obs is 2D (batch, features)
                if obs.dim() == 1:
                    obs = obs.unsqueeze(0)
                elif obs.dim() == 0:
                    obs = obs.view(1, 1)
                # If obs is 3D (batch, 1, 1), flatten to (batch, 1)
                if obs.dim() == 3 and obs.shape[1] == 1 and obs.shape[2] == 1:
                    obs = obs.view(obs.shape[0], 1)
                encoded = extractor(obs)
                # If output is 2D (batch, features), flatten if batch==1
                if encoded.dim() == 2 and encoded.shape[0] == 1:
                    encoded = encoded.squeeze(0)
            else:
                # For Graph, pass the GraphInstance directly
                encoded = extractor(obs)
            encoded_tensor_list.append(encoded)
        return th.cat(encoded_tensor_list, dim=0)


@pytest.fixture
def graph_env():
    """Create a GraphEnv instance for testing."""
    return GraphEnv()


@pytest.fixture
def graph_only_env():
    """Create a GraphEnv instance with only graph observations for testing."""
    return GraphEnv(include_box_obs=False)



@pytest.mark.parametrize(
    "include_box_obs,features_dim",
    [
        (True, 10),   # Dict space with Graph and Box
        (False, 8),   # Dict space with only Graph
    ]
)
def test_feature_extractor(include_box_obs, features_dim):
    """Test that a custom feature extractor can handle Graph spaces."""
    env = GraphEnv(include_box_obs=include_box_obs)
    
    # Create and initialize the feature extractor
    extractor = MyExtractor(env.observation_space, features_dim=features_dim)
    
    # Get a sample observation
    obs, _ = env.reset()
    
    # Process the observation
    features = extractor(obs)
    
    # Calculate expected output size based on number of components
    expected_dim = features_dim * len(env.observation_space.spaces)
    
    # Check the output shape
    assert features.shape == (expected_dim,)


def test_get_obs_shape_for_graph_space():
    """Test that get_obs_shape returns correct shapes for Graph spaces."""
    env = GraphEnv()
    
    # Check shape for the graph space
    graph_shape = get_obs_shape(env.observation_space.spaces["my_graph"])
    assert graph_shape == ("graph",), "Graph space shape should be ('graph',) for buffer allocation"
    
    # Check shape for the box space
    box_shape = get_obs_shape(env.observation_space.spaces["my_box"])
    assert box_shape == (1,), "Box space shape should be (1,)"


@pytest.mark.parametrize(
    "features_dim", [5, 10, 20]
)
def test_custom_feature_extractor_dimensions(features_dim):
    """
    Test that PPO can be initialized with different feature extractor dimensions.
    
    :param features_dim: Dimension of the features extracted by the custom extractor
    """
    env = GraphEnv()
    policy_kwargs = dict(
        features_extractor_class=MyExtractor,
        features_extractor_kwargs=dict(features_dim=features_dim),
        share_features_extractor=False,
    )
    
    # Test model initialization
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs)
    assert model.policy is not None


@pytest.mark.parametrize(
    "share_extractor", [True, False]
)
def test_shared_feature_extractor(share_extractor):
    """
    Test that PPO can be initialized with shared or separate feature extractors.
    
    :param share_extractor: Whether to share the feature extractor between actor and critic
    """
    env = GraphEnv()
    policy_kwargs = dict(
        features_extractor_class=MyExtractor,
        share_features_extractor=share_extractor,
    )
    
    # Test model initialization
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs)
    assert model.policy is not None


if __name__ == "__main__":
    # Run the tests directly
    env = GraphEnv()
    test_env_checkers_support_graph_space(env)
    test_feature_extractor(True, 10)
    test_feature_extractor(False, 8)
    test_get_obs_shape_for_graph_space()
    test_custom_feature_extractor_dimensions(10)
    test_shared_feature_extractor(True)
    test_shared_feature_extractor(False)
    print("All tests passed!")
