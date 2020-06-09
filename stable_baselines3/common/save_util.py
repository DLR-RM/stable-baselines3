"""
Save util taken from stable_baselines
used to serialize data (class parameters) of model classes
"""
import os
import io
import json
import base64
import functools
from typing import Dict, Any, Tuple, Optional
import warnings
import zipfile

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

    return functools.reduce(_getattr, [obj] + attr.split('.'))


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
    pre, _, post = attr.rpartition('.')
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
            base64_encoded = base64.b64encode(
                cloudpickle.dumps(data_item)
            ).decode()

            # Use ":" to make sure we do
            # not override these keys
            # when we include variables of the object later
            cloudpickle_serialization = {
                ":type:": str(type(data_item)),
                ":serialized:": base64_encoded
            }

            # Add first-level JSON-serializable items of the
            # object for further details (but not deeper than this to
            # avoid deep nesting).
            # First we check that object has attributes (not all do,
            # e.g. numpy scalars)
            if hasattr(data_item, "__dict__") or isinstance(data_item, dict):
                # Take elements from __dict__ for custom classes
                item_generator = (
                    data_item.items if isinstance(data_item, dict) else data_item.__dict__.items
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


def json_to_data(json_string: str,
                 custom_objects: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
                warnings.warn(f"Could not deserialize object {data_key}. " +
                              "Consider using `custom_objects` argument to replace " +
                              "this object.")
            return_data[data_key] = deserialized_object
        else:
            # Read as it is
            return_data[data_key] = data_item
    return return_data


def save_to_zip_file(save_path: str, data: Dict[str, Any] = None,
                     params: Dict[str, Any] = None, tensors: Dict[str, Any] = None) -> None:
    """
    Save a model to a zip archive.

    :param save_path: Where to store the model.
    :param data: Class parameters being stored.
    :param params: Model parameters being stored expected to contain an entry for every
                   state_dict with its name and the state_dict.
    :param tensors: Extra tensor variables expected to contain name and value of tensors
    """

    # data/params can be None, so do not
    # try to serialize them blindly
    if data is not None:
        serialized_data = data_to_json(data)

    # Check postfix if save_path is a string
    if isinstance(save_path, str):
        _, ext = os.path.splitext(save_path)
        if ext == "":
            save_path += ".zip"

    # Create a zip-archive and write our objects
    # there. This works when save_path is either
    # str or a file-like
    with zipfile.ZipFile(save_path, "w") as archive:
        # Do not try to save "None" elements
        if data is not None:
            archive.writestr("data", serialized_data)
        if tensors is not None:
            with archive.open('tensors.pth', mode="w") as tensors_file:
                th.save(tensors, tensors_file)
        if params is not None:
            for file_name, dict_ in params.items():
                with archive.open(file_name + '.pth', mode="w") as param_file:
                    th.save(dict_, param_file)


def load_from_zip_file(load_path: str, load_data: bool = True) -> (Tuple[Optional[Dict[str, Any]],
                                                                   Optional[TensorDict],
                                                                   Optional[TensorDict]]):
    """
    Load model data from a .zip archive

    :param load_path: Where to load the model from
    :param load_data: Whether we should load and return data
        (class parameters). Mainly used by 'load_parameters' to only load model parameters (weights)
    :return: (dict),(dict),(dict) Class parameters, model state_dicts (dict of state_dict)
        and dict of extra tensors
    """
    # Check if file exists if load_path is a string
    if isinstance(load_path, str):
        if not os.path.exists(load_path):
            if os.path.exists(load_path + ".zip"):
                load_path += ".zip"
            else:
                raise ValueError(f"Error: the file {load_path} could not be found")

    # set device to cpu if cuda is not available
    device = get_device()

    # Open the zip archive and load data
    try:
        with zipfile.ZipFile(load_path, "r") as archive:
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
                with archive.open('tensors.pth', mode="r") as tensor_file:
                    # File has to be seekable, but opt_param_file is not, so load in BytesIO first
                    # fixed in python >= 3.7
                    file_content = io.BytesIO()
                    file_content.write(tensor_file.read())
                    # go to start of file
                    file_content.seek(0)
                    # load the parameters with the right ``map_location``
                    tensors = th.load(file_content, map_location=device)

            # check for all other .pth files
            other_files = [file_name for file_name in namelist if
                           os.path.splitext(file_name)[1] == ".pth" and file_name != "tensors.pth"]
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
                        params[os.path.splitext(file_path)[0]] = th.load(file_content, map_location=device)
    except zipfile.BadZipFile:
        # load_path wasn't a zip file
        raise ValueError(f"Error: the file {load_path} wasn't a zip-file")
    return data, params, tensors
