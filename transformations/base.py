from abc import abstractmethod, ABC
import os
from typing import Optional

import joblib

import numpy as np
from sklearn.base import TransformerMixin


class ExtendedSaveLoadPickle:

    @staticmethod
    def save_base_model(model, file_name, file_path: Optional[str] = None):
        complete_file_path = file_name
        if file_path:
            complete_file_path = os.path.join(file_path, file_name)
        joblib.dump(model, complete_file_path)

    @staticmethod
    def load_base_model(file_name, file_path: Optional[str] = None):
        complete_file_path = file_name
        if file_path:
            complete_file_path = os.path.join(file_path, file_name)
        model = joblib.load(complete_file_path)
        return model


class TargetedTransformer(TransformerMixin, ExtendedSaveLoadPickle):
    """
    Dummy class that allows us to modify only the methods that interest us,
    avoiding redundancy.
    """

    def __init__(self, columns_to_apply=None, **model_kwargs):
        self._columns_to_apply = columns_to_apply
        self._model_kwargs = model_kwargs
        self._array_column_names = []
        self._models = {}

    def _get_features(self, x):
        return x.columns if self._columns_to_apply is None else self._columns_to_apply

    def fit(self, x, y=None):
        return self

    @abstractmethod
    def transform(self, x, y=None):
        pass

    def save_models(self,  models_to_save: Optional[list] = None, file_path: Optional[str] = None):
        """
        Saves the models included in the models_to_save in a pkl at file_path
        :param file_path: path where the models will be saved.
        If file_path is None it will be saved at the working directory
        :param models_to_save: if None it saves all the models. Else, it must contain a list with the names of the
        individual models in self._encoders, that will be the name by default of the model saved.
        """
        if models_to_save:
            for model_name in models_to_save:
                model = self._models[model_name]
                file_name = model_name + '.pkl'
                self.save_base_model(model, file_name, file_path)
        else:
            for model_name, model in self._models.items():
                file_name = model_name + '.pkl'
                self.save_base_model(model, file_name, file_path)

    def load_models(self, models_to_load: Optional[list] = None, file_path: Optional[str] = None):
        """
        Loads the models at the instance, specified in models_to_load that are in file_path
        :param file_path: path where the models are. If None, it will be the working directory
        :param models_to_load: name of the files to load. If None, it will load all the pkl files at the file_path.
        :return: dictionary with the models loaded
        """
        file_names = models_to_load
        if models_to_load is None:
            if file_path is None:
                file_path = os.getcwd()
            file_names = [file for file in os.listdir(file_path) if file.endswith('pkl')]

        for file_name in file_names:
            model = self.load_base_model(file_name, file_path)
            model_name = file_name.replace('.pkl', '')
            self._models[model_name] = model
            self._array_column_names.extend(self._get_array_column_names_from_model(model, model_name))
        return self._models

    @staticmethod
    def _get_array_column_names_from_model(model: Optional = None, model_name: str = ''):
        return [model_name]


class SklearnTargetedTransformer(TargetedTransformer, ABC):

    def __init__(self, columns_to_apply=None, **model_kwargs):
        super().__init__(columns_to_apply, **model_kwargs)

    def fit(self, x, y=None):
        self._array_column_names = []
        self._encoders = {}
        features = self._get_features(x)

        for feature in features:
            self._encoders[feature] = self._individual_fit(x, feature)
        return self

    @abstractmethod
    def _individual_fit(self, x, feature):
        pass

    def transform(self, x, y=None):
        features = self._get_features(x)
        transformed_features = [self._individual_transform(x, feature) for feature in features]
        transformed_features = np.concatenate(transformed_features, axis=1)
        return transformed_features

    @abstractmethod
    def _individual_transform(self, x, feature):
        pass


class NoTransformer(TargetedTransformer):

    def fit(self, x, y=None):
        features = self._get_features(x)
        self._array_column_names.extend(features)
        return self

    def transform(self, x, y=None):
        features = self._get_features(x)
        return x[features]

    def save_models(self,  models_to_save: Optional[list] = None, file_path: Optional[str] = None):
        pass

    def load_models(self, models_to_load: Optional[list] = None, file_path: Optional[str] = None):
        pass
