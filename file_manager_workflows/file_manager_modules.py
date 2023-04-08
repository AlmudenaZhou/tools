import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

import joblib
import yaml


class Manager(ABC):

    _extension: str

    @classmethod
    def get_complete_file_path(cls, raw_file_name, file_path):
        file_name = raw_file_name + cls._extension
        complete_file_path = file_name
        if file_path:
            complete_file_path = os.path.join(file_path, file_name)
        return complete_file_path

    @staticmethod
    def create_file_path(file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            logging.info(f'{file_path} was created')

    def save(self, obj, raw_file_name, file_path: Optional[str] = None):
        complete_file_path = self.get_complete_file_path(raw_file_name, file_path)
        self.create_file_path(file_path)
        self._specific_save(obj, complete_file_path)
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
    def _specific_load(complete_file_path):
        pass


class PickleManager(Manager):

    _extension = '.pkl'

    @staticmethod
    def _specific_save(obj, complete_file_path):
        joblib.dump(obj, complete_file_path)

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
    def _specific_load(complete_file_path):
        with open(complete_file_path, "r") as file:
            obj = yaml.safe_load(file)
        return obj
