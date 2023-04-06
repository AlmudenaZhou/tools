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
        self._encoders = {}

    def _get_features(self, x):
        return x.columns if self._columns_to_apply is None else self._columns_to_apply

    def fit(self, x, y=None):
        return self

    @abstractmethod
    def transform(self, x, y=None):
        pass


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
