import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import yaml


class Manager(ABC):

    _extension: str

    @classmethod
    def get_complete_file_path(cls, raw_file_name, file_path):
        """
        Function that returns the raw_file_name with the specific extension appended and joined to the file_path.
        :param raw_file_name: file_name without the extension
        :param file_path: file_path. If None, it uses the working directory
        :return:
        """
        file_name = raw_file_name + cls._extension
        if file_path is None:
            file_path = os.getcwd()
        complete_file_path = os.path.join(file_path, file_name)
        return complete_file_path

    @staticmethod
    def create_file_path(file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            logging.info(f'{file_path} was created')

    def save(self, obj, raw_file_name, file_path: Optional[str] = None, append_to_file=False):
        complete_file_path = self.get_complete_file_path(raw_file_name, file_path)

        if file_path is not None:
            self.create_file_path(file_path)

        if not append_to_file or not os.path.exists(complete_file_path):
            self._specific_save(obj, complete_file_path)
        else:
            self._specific_append(obj, complete_file_path)
        logging.info(f'{complete_file_path} was saved')

    def load(self, raw_file_name, file_path: Optional[str] = None):
        complete_file_path = self.get_complete_file_path(raw_file_name, file_path)
        obj = self._specific_load(complete_file_path)
        return obj

    @staticmethod
    @abstractmethod
    def _specific_save(obj, complete_file_path):
        pass

    @staticmethod
    @abstractmethod
    def _specific_append(obj, complete_file_path):
        pass

    @staticmethod
    @abstractmethod
    def _specific_load(complete_file_path):
        pass


class PickleManager(Manager):

    _extension = '.pkl'

    @staticmethod
    def _specific_save(obj, complete_file_path):
        joblib.dump(obj, complete_file_path)

    @staticmethod
    def _specific_append(obj, complete_file_path):
        prev_obj = PickleManager._specific_load(complete_file_path)
        PickleManager._check_same_type_obj_specific_append(prev_obj, obj, complete_file_path)
        new_obj = PickleManager._append_objects(prev_obj, obj)
        joblib.dump(new_obj, complete_file_path)

    @staticmethod
    def _append_objects(prev_obj, obj):

        if isinstance(prev_obj, list):
            new_obj = prev_obj.extend(obj)

        elif isinstance(prev_obj, tuple):
            new_obj = prev_obj + obj

        elif isinstance(prev_obj, dict):
            prev_obj.update(obj)
            new_obj = prev_obj

        elif isinstance(prev_obj, np.ndarray):
            new_obj = np.concatenate([prev_obj, obj])

        elif isinstance(prev_obj, (pd.Series, pd.DataFrame)):
            new_obj = pd.concat([prev_obj, obj])

        else:
            raise NotImplemented(f'The type {type(prev_obj)} is not implemented as an append method of pickle manager')

        return new_obj

    @staticmethod
    def _check_same_type_obj_specific_append(prev_obj, obj, complete_file_path):
        prev_obj_type = type(prev_obj)
        obj_type = type(obj)
        error_msg = f'The object at {complete_file_path} is type {prev_obj_type} while the new one you are ' \
                    f'passing is {obj_type}. Both must be the same to append it one to another.'
        if prev_obj_type != obj_type:
            raise TypeError(error_msg)

    @staticmethod
    def _specific_load(complete_file_path):
        obj = joblib.load(complete_file_path)
        return obj


class YamlManager(Manager):

    _extension = '.yaml'

    @staticmethod
    def _specific_save(obj, complete_file_path):
        with open(complete_file_path, "w") as file:
            yaml.dump(obj, file)

    @staticmethod
    def _specific_append(obj, complete_file_path):
        prev_obj = YamlManager._specific_load(complete_file_path)
        prev_obj.update(obj)
        YamlManager._specific_save(prev_obj, complete_file_path)

    @staticmethod
    def _specific_load(complete_file_path):
        with open(complete_file_path, "r") as file:
            obj = yaml.safe_load(file)
        return obj
