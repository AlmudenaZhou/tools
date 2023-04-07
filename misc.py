import importlib


def import_module_from_name(module_name, class_name, **class_kwargs):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    instance = class_(**class_kwargs)
    return instance
