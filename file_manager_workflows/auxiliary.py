import hashlib
import pickle


def hash_object(obj):
    # Convert the object to a bytes string using pickle
    obj_bytes = pickle.dumps(obj)

    # Generate a SHA-256 hash of the object bytes
    hash_obj = hashlib.sha256(obj_bytes)
    return hash_obj.hexdigest()
