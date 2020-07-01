"""
Save util taken from stable_baselines
used to serialize data (class parameters) of model classes
"""
import io
import os
import json
import base64
import functools
from typing import Dict, Any, Tuple, Optional, Union
import warnings
import zipfile
import pathlib
import pickle

import torch as th
import cloudpickle

from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device


def recursive_getattr(obj: Any, attr: str, *args) -> Any:
    """
    Recursive version of getattr
    taken from https://stackoverflow.com/questions/31174295

    Ex:
    > MyObject.sub_object = SubObject(name='test')
    > recursive_getattr(MyObject, 'sub_object.name')  # return test
    :param obj: (Any)
    :param attr: (str) Attribute to retrieve
    :return: (Any) The attribute
    """

    def _getattr(obj: Any, attr: str) -> Any:
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def recursive_setattr(obj: Any, attr: str, val: Any) -> None:
    """
    Recursive version of setattr
    taken from https://stackoverflow.com/questions/31174295

    Ex:
    > MyObject.sub_object = SubObject(name='test')
    > recursive_setattr(MyObject, 'sub_object.name', 'hello')
    :param obj: (Any)
    :param attr: (str) Attribute to set
    :param val: (Any) New value of the attribute
    """
    pre, _, post = attr.rpartition(".")
    return setattr(recursive_getattr(obj, pre) if pre else obj, post, val)


def is_json_serializable(item: Any) -> bool:
    """
    Test if an object is serializable into JSON

    :param item: (object) The object to be tested for JSON serialization.
    :return: (bool) True if object is JSON serializable, false otherwise.
    """
    # Try with try-except struct.
    json_serializable = True
    try:
        _ = json.dumps(item)
    except TypeError:
        json_serializable = False
    return json_serializable


def data_to_json(data: Dict[str, Any]) -> str:
    """
    Turn data (class parameters) into a JSON string for storing

    :param data: (Dict[str, Any]) Dictionary of class parameters to be
        stored. Items that are not JSON serializable will be
        pickled with Cloudpickle and stored as bytearray in
        the JSON file
    :return: (str) JSON string of the data serialized.
    """
    # First, check what elements can not be JSONfied,
    # and turn them into byte-strings
    serializable_data = {}
    for data_key, data_item in data.items():
        # See if object is JSON serializable
        if is_json_serializable(data_item):
            # All good, store as it is
            serializable_data[data_key] = data_item
        else:
            # Not serializable, cloudpickle it into
            # bytes and convert to base64 string for storing.
            # Also store type of the class for consumption
            # from other languages/humans, so we have an
            # idea what was being stored.
            base64_encoded = base64.b64encode(cloudpickle.dumps(data_item)).decode()

            # Use ":" to make sure we do
            # not override these keys
            # when we include variables of the object later
            cloudpickle_serialization = {
                ":type:": str(type(data_item)),
                ":serialized:": base64_encoded,
            }

            # Add first-level JSON-serializable items of the
            # object for further details (but not deeper than this to
            # avoid deep nesting).
            # First we check that object has attributes (not all do,
            # e.g. numpy scalars)
            if hasattr(data_item, "__dict__") or isinstance(data_item, dict):
                # Take elements from __dict__ for custom classes
                item_generator = (
                    data_item.items
                    if isinstance(data_item, dict)
                    else data_item.__dict__.items
                )
                for variable_name, variable_item in item_generator():
                    # Check if serializable. If not, just include the
                    # string-representation of the object.
                    if is_json_serializable(variable_item):
                        cloudpickle_serialization[variable_name] = variable_item
                    else:
                        cloudpickle_serialization[variable_name] = str(variable_item)

            serializable_data[data_key] = cloudpickle_serialization
    json_string = json.dumps(serializable_data, indent=4)
    return json_string


def json_to_data(
    json_string: str, custom_objects: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Turn JSON serialization of class-parameters back into dictionary.

    :param json_string: (str) JSON serialization of the class-parameters
        that should be loaded.
    :param custom_objects: (dict) Dictionary of objects to replace
        upon loading. If a variable is present in this dictionary as a
        key, it will not be deserialized and the corresponding item
        will be used instead. Similar to custom_objects in
        `keras.models.load_model`. Useful when you have an object in
        file that can not be deserialized.
    :return: (dict) Loaded class parameters.
    """
    if custom_objects is not None and not isinstance(custom_objects, dict):
        raise ValueError("custom_objects argument must be a dict or None")

    json_dict = json.loads(json_string)
    # This will be filled with deserialized data
    return_data = {}
    for data_key, data_item in json_dict.items():
        if custom_objects is not None and data_key in custom_objects.keys():
            # If item is provided in custom_objects, replace
            # the one from JSON with the one in custom_objects
            return_data[data_key] = custom_objects[data_key]
        elif isinstance(data_item, dict) and ":serialized:" in data_item.keys():
            # If item is dictionary with ":serialized:"
            # key, this means it is serialized with cloudpickle.
            serialization = data_item[":serialized:"]
            # Try-except deserialization in case we run into
            # errors. If so, we can tell bit more information to
            # user.
            try:
                base64_object = base64.b64decode(serialization.encode())
                deserialized_object = cloudpickle.loads(base64_object)
            except RuntimeError:
                warnings.warn(
                    f"Could not deserialize object {data_key}. "
                    + "Consider using `custom_objects` argument to replace "
                    + "this object."
                )
            return_data[data_key] = deserialized_object
        else:
            # Read as it is
            return_data[data_key] = data_item
    return return_data


@functools.singledispatch
def open_path(
    path: Union[str, pathlib.Path, io.BufferedIOBase], mode: str, verbose=0, suffix=None
):
    """
    Opens a path for reading or writing with a preferred suffix and raises debug information.
    If the provided path is a derivative of io.BufferedIOBase it ensures that the file
    matches the provided mode, i.e. If the mode is read ("r", "read") it checks that the path is readable.
    If the mode is write ("w", "write") it checks that the file is writable.

    If the provided path is a string or a pathlib.Path, it ensures that it exists. If the mode is "read"
    it checks that it exists, if it doesn't exist it attempts to read path.suffix if a suffix is provided.
    If the mode is "write" and the path does not exist, it creates all the parent folders. If the path
    points to a folder, it changes the path to path_2. If the path already exists and verbose == 2,
    it raises a warning.

    :param path: (Union[str, pathlib.Path, io.BufferedIOBase]) the path to open.
        if save_path is a str or pathlib.Path and mode is "w", single dispatch ensures that the
        path actually exists. If path is a io.BufferedIOBase the path exists.
    :param mode: (str) how to open the file. "w"|"write" for writing, "r"|"read" for reading.
    :param verbose: (int) Verbosity level, 0 means only warnings, 2 means debug information.
    :param suffix: (str) The preferred suffix. If mode is "w" then the opened file has the suffix.
        If mode is "r" then we attempt to open the path. If an error is raised and the suffix
        is not None, we attempt to open the path with the suffix.
    """
    if not isinstance(path, io.BufferedIOBase):
        raise TypeError("Path parameter has invalid type.", io.BufferedIOBase)
    if path.closed:
        raise ValueError("File stream is closed.")
    mode = mode.lower()
    try:
        mode = {"write": "w", "read": "r", "w": "w", "r": "r"}[mode]
    except KeyError:
        raise ValueError("Expected mode to be either 'w' or 'r'.")
    if ("w" == mode) and not path.writable() or ("r" == mode) and not path.readable():
        e1 = "writable" if "w" == mode else "readable"
        raise ValueError(f"Expected a {e1} file.")
    return path


@open_path.register(str)
def open_path_str(
    path: str, mode: str, verbose=0, suffix=None
) -> io.BufferedIOBase:
    """
    Open a path given by a string. If writing to the path, the function ensures
    that the path exists.

    :param path: (str) the path to open. If mode is "w" then it ensures that the path exists
        by creating the necessary folders and renaming path if it points to a folder.
    :param mode: (str) how to open the file. "w" for writing, "r" for reading.
    :param verbose: (int) Verbosity level, 0 means only warnings, 2 means debug information.
    :param suffix: (str) The preferred suffix. If mode is "w" then the opened file has the suffix.
        If mode is "r" then we attempt to open the path. If an error is raised and the suffix
        is not None, we attempt to open the path with the suffix.
    """
    return open_path(pathlib.Path(path), mode, verbose, suffix)


@open_path.register(pathlib.Path)
def open_path_pathlib(
    path: pathlib.Path, mode: str, verbose=0, suffix=None
) -> io.BufferedIOBase:
    """
    Open a path given by a string. If writing to the path, the function ensures
    that the path exists.

    :param path: (pathlib.Path) the path to check. If mode is "w" then it
        ensures that the path exists by creating the necessary folders and
        renaming path if it points to a folder.
    :param mode: (str) how to open the file. "w" for writing, "r" for reading.
    :param verbose: (int) Verbosity level, 0 means only warnings, 2 means debug information.
    :param suffix: (str) The preferred suffix. If mode is "w" then the opened file has the suffix.
        If mode is "r" then we attempt to open the path. If an error is raised and the suffix
        is not None, we attempt to open the path with the suffix.
    """
    if mode not in ("w", "r"):
        raise ValueError("Expected mode to be either 'w' or 'r'.")

    if mode == "r":
        try:
            path = path.open("rb")
        except FileNotFoundError as error:
            if suffix is not None and suffix != "":
                newpath = pathlib.Path(f"{path}.{suffix}")
                if verbose == 2:
                    warnings.warn(f"Path '{path}' not found. Attempting {newpath}.")
                path, suffix = newpath, None
            else:
                raise error
    else:
        try:
            if path.suffix == "" and suffix is not None and suffix != "":
                path = pathlib.Path(f"{path}.{suffix}")
            if path.exists() and path.is_file() and verbose == 2:
                warnings.warn(f"Path '{path}' exists, will overwrite it.")
            path = path.open("wb")
        except IsADirectoryError:
            warnings.warn(f"Path '{path}' is a folder. Will save instead to {path}_2")
            path = pathlib.Path(f"{path}_2")
        except FileNotFoundError:  # Occurs when the parent folder doesn't exist
            warnings.warn(f"Path '{path.parent}' does not exist. Will create it.")
            path.parent.mkdir(exist_ok=True, parents=True)

    # if opening was successful uses the identity function
    # if opening failed with IsADirectory|FileNotFound, calls open_path_pathlib
    #   with corrections
    # if reading failed with FileNotFoundError, calls open_path_pathlib with suffix

    return open_path(path, mode, verbose, suffix)


def save_to_zip_file(
    save_path: Union[str, pathlib.Path, io.BufferedIOBase],
    data: Dict[str, Any] = None,
    params: Dict[str, Any] = None,
    tensors: Dict[str, Any] = None,
    verbose=0,
) -> None:
    """
    Save a model to a zip archive.

    :param save_path: (Union[str, pathlib.Path, io.BufferedIOBase]) Where to store the model.
        if save_path is a str or pathlib.Path ensures that the path actually exists.
    :param data: Class parameters being stored.
    :param params: Model parameters being stored expected to contain an entry for every
                   state_dict with its name and the state_dict.
    :param tensors: Extra tensor variables expected to contain name and value of tensors
    :param verbose: (int) Verbosity level, 0 means only warnings, 2 means debug information
    """

    save_path = open_path(save_path, "w", verbose=0, suffix="zip")
    # data/params can be None, so do not
    # try to serialize them blindly
    if data is not None:
        serialized_data = data_to_json(data)

    # Create a zip-archive and write our objects there.
    with zipfile.ZipFile(save_path, mode="w") as archive:
        # Do not try to save "None" elements
        if data is not None:
            archive.writestr("data", serialized_data)
        if tensors is not None:
            with archive.open("tensors.pth", mode="w") as tensors_file:
                th.save(tensors, tensors_file)
        if params is not None:
            for file_name, dict_ in params.items():
                with archive.open(file_name + ".pth", mode="w") as param_file:
                    th.save(dict_, param_file)


def save_to_pkl(
    path: Union[str, pathlib.Path, io.BufferedIOBase], obj, verbose=0
) -> None:
    """
    Save an object to path creating the necessary folders along the way.
    If the path exists and is a directory, it will raise a warning and rename the path.
    If a suffix is provided in the path, it will use that suffix, otherwise, it will use '.pkl'.

    :param path: (Union[str, pathlib.Path, io.BufferedIOBase]) the path to open.
        if save_path is a str or pathlib.Path and mode is "w", single dispatch ensures that the
        path actually exists. If path is a io.BufferedIOBase the path exists.
    :param obj: The object to save.
    :param verbose: (int) Verbosity level, 0 means only warnings, 2 means debug information.
    """
    with open_path(path, "w", verbose=verbose, suffix="pkl") as file_handler:
        pickle.dump(obj, file_handler)


def load_from_pkl(path: Union[str, pathlib.Path, io.BufferedIOBase], verbose=0) -> Any:
    """
    Load an object from the path. If a suffix is provided in the path, it will use that suffix.
    If the path does not exist, it will attempt to load using the .pkl suffix.

    :param path: (Union[str, pathlib.Path, io.BufferedIOBase]) the path to open.
        if save_path is a str or pathlib.Path and mode is "w", single dispatch ensures that the
        path actually exists. If path is a io.BufferedIOBase the path exists.
    :param verbose: (int) Verbosity level, 0 means only warnings, 2 means debug information.
    """
    with open_path(path, "r", verbose=verbose, suffix="pkl") as file_handler:
        return pickle.load(file_handler)


def load_from_zip_file(
    load_path: Union[str, pathlib.Path, io.BufferedIOBase],
    load_data: bool = True,
    verbose=0,
) -> (Tuple[Optional[Dict[str, Any]], Optional[TensorDict], Optional[TensorDict]]):
    """
    Load model data from a .zip archive

    :param load_path: (str, pathlib.Path, io.BufferedIOBase) Where to load the model from
    :param load_data: Whether we should load and return data
        (class parameters). Mainly used by 'load_parameters' to only load model parameters (weights)
    :return: (dict),(dict),(dict) Class parameters, model state_dicts (dict of state_dict)
        and dict of extra tensors
    """
    load_path = open_path(load_path, "r", verbose=verbose, suffix="zip")

    # set device to cpu if cuda is not available
    device = get_device()

    # Open the zip archive and load data
    try:
        with zipfile.ZipFile(load_path) as archive:
            namelist = archive.namelist()
            # If data or parameters is not in the
            # zip archive, assume they were stored
            # as None (_save_to_file_zip allows this).
            data = None
            tensors = None
            params = {}

            if "data" in namelist and load_data:
                # Load class parameters and convert to string
                json_data = archive.read("data").decode()
                data = json_to_data(json_data)

            if "tensors.pth" in namelist and load_data:
                # Load extra tensors
                with archive.open("tensors.pth", mode="r") as tensor_file:
                    # File has to be seekable, but opt_param_file is not, so load in BytesIO first
                    # fixed in python >= 3.7
                    file_content = io.BytesIO()
                    file_content.write(tensor_file.read())
                    # go to start of file
                    file_content.seek(0)
                    # load the parameters with the right ``map_location``
                    tensors = th.load(file_content, map_location=device)

            # check for all other .pth files
            other_files = [
                file_name
                for file_name in namelist
                if os.path.splitext(file_name)[1] == ".pth"
                and file_name != "tensors.pth"
            ]
            # if there are any other files which end with .pth and aren't "params.pth"
            # assume that they each are optimizer parameters
            if len(other_files) > 0:
                for file_path in other_files:
                    with archive.open(file_path, mode="r") as opt_param_file:
                        # File has to be seekable, but opt_param_file is not, so load in BytesIO first
                        # fixed in python >= 3.7
                        file_content = io.BytesIO()
                        file_content.write(opt_param_file.read())
                        # go to start of file
                        file_content.seek(0)
                        # load the parameters with the right ``map_location``
                        params[os.path.splitext(file_path)[0]] = th.load(
                            file_content, map_location=device
                        )
    except zipfile.BadZipFile:
        # load_path wasn't a zip file
        raise ValueError(f"Error: the file {load_path} wasn't a zip-file")
    return data, params, tensors
