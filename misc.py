import importlib
import hashlib
import pickle


def instance_class_from_module_and_name(module_name, class_name, **class_kwargs):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    instance = class_(**class_kwargs)
    return instance


def hash_object(obj):
    # Convert the object to a bytes string using pickle
    obj_bytes = pickle.dumps(obj)

    # Generate a SHA-256 hash of the object bytes
    hash_obj = hashlib.sha256(obj_bytes)
    return hash_obj.hexdigest()
