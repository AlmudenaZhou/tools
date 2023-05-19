import importlib
import re
import hashlib
import pickle


def instance_class_from_module_and_name(module_name, class_name, **class_kwargs):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    instance = class_(**class_kwargs)
    return instance


def hash_object_with_sha256(obj):
    # Convert the object to a bytes string using pickle
    obj_bytes = pickle.dumps(obj)

    # Generate a SHA-256 hash of the object bytes
    hash_obj = hashlib.sha256(obj_bytes)
    return hash_obj.hexdigest()


def pascal_to_snake_case(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('__([A-Z])', r'_\1', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()
