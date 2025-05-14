"""
Helpers for dealing with vectorized environments.
"""

from typing import Any

import numpy as np
from gymnasium import spaces

from stable_baselines3.common.preprocessing import check_for_nested_spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs


def dict_to_obs(obs_space: spaces.Space, obs_dict: dict[Any, np.ndarray]) -> VecEnvObs:
    """
    Convert an internal representation raw_obs into the appropriate type
    specified by space.

    :param obs_space: an observation space.
    :param obs_dict: a dict of numpy arrays.
    :return: returns an observation of the same type as space.
        If space is Dict, function is identity; if space is Tuple, converts dict to Tuple;
        otherwise, space is unstructured and returns the value raw_obs[None].
    """
    if isinstance(obs_space, spaces.Dict):
        # For Dict spaces, if a subspace is a Graph, handle both gymnasium.spaces.Graph and gymnasium.spaces.graph.Graph
        result = {}
        for k, v in obs_dict.items():
            subspace = obs_space.spaces[k]
            # Support both gymnasium.spaces.Graph and gymnasium.spaces.graph.Graph
            is_graph = False
            try:
                import gymnasium.spaces.graph as sp_graph
                is_graph = isinstance(subspace, (spaces.Graph, getattr(sp_graph, "Graph", type(None))))
            except Exception:
                is_graph = isinstance(subspace, spaces.Graph)
            if is_graph:
                # If v is a list (batched env), extract the first element (SB3 expects a single GraphInstance)
                if isinstance(v, list):
                    # If the list contains GraphInstance, just return the first one (for single env)
                    if len(v) == 1 and hasattr(v[0], "nodes") and hasattr(v[0], "edges") and hasattr(v[0], "edge_links"):
                        result[k] = v[0]
                    else:
                        # If the list contains dicts (from env_checker), convert dict to GraphInstance
                        if len(v) == 1 and isinstance(v[0], dict) and {"nodes", "edges", "edge_links"}.issubset(v[0].keys()):
                            from gymnasium.spaces.graph import GraphInstance
                            result[k] = GraphInstance(**v[0])
                        else:
                            raise ValueError(f"Expected a single GraphInstance or dict in list for key={k}, got {v}.")
                elif isinstance(v, dict) and {"nodes", "edges", "edge_links"}.issubset(v.keys()):
                    # If v is a dict (from env_checker), convert to GraphInstance
                    from gymnasium.spaces.graph import GraphInstance
                    result[k] = GraphInstance(**v)
                else:
                    result[k] = v
            else:
                result[k] = v
        return result
    elif isinstance(obs_space, spaces.Tuple):
        assert len(obs_dict) == len(obs_space.spaces), "size of observation does not match size of observation space"
        return tuple(obs_dict[i] for i in range(len(obs_space.spaces)))
    else:
        assert set(obs_dict.keys()) == {None}, "multiple observation keys for unstructured observation space"
        return obs_dict[None]


def obs_space_info(obs_space: spaces.Space) -> tuple[list[str], dict[Any, tuple[int, ...]], dict[Any, np.dtype]]:
    """
    Get dict-structured information about a gym.Space.

    Dict spaces are represented directly by their dict of subspaces.
    Tuple spaces are converted into a dict with keys indexing into the tuple.
    Unstructured spaces are represented by {None: obs_space}.

    :param obs_space: an observation space
    :return: A tuple (keys, shapes, dtypes):
        keys: a list of dict keys.
        shapes: a dict mapping keys to shapes.
        dtypes: a dict mapping keys to dtypes.
    """
    check_for_nested_spaces(obs_space)
    if isinstance(obs_space, spaces.Dict):
        assert isinstance(obs_space.spaces, dict), "Dict space must have ordered subspaces"
        subspaces = obs_space.spaces
    elif isinstance(obs_space, spaces.Tuple):
        subspaces = {i: space for i, space in enumerate(obs_space.spaces)}  # type: ignore[assignment,misc]
    else:
        assert not hasattr(obs_space, "spaces"), f"Unsupported structured space '{type(obs_space)}'"
        subspaces = {None: obs_space}  # type: ignore[assignment,dict-item]
    keys = []
    shapes = {}
    dtypes = {}
    for key, box in subspaces.items():
        keys.append(key)
        if isinstance(box, spaces.Graph):
            # For graph spaces, we don't have a fixed shape
            # Store None as the shape to indicate special handling is needed
            shapes[key] = None
            dtypes[key] = np.float32  # Default dtype for graph data
        else:
            shapes[key] = box.shape
            dtypes[key] = box.dtype
    return keys, shapes, dtypes  # type: ignore[return-value]
