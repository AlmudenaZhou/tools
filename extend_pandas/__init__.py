import os
import pathlib
import importlib


desktop = pathlib.Path(os.path.dirname(__file__))

modules = list(desktop.rglob("*.py"))

for module in modules:
    if os.path.isfile(module) and not module.parts[-1] == '__init__.py':
        importlib.import_module('.' + os.path.relpath(module, os.path.dirname(__file__))[:-3].replace('\\', '.'),
                                package='tools.extend_pandas')
