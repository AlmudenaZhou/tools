from abc import abstractmethod
from typing import Optional

import numpy as np
from sklearn.base import TransformerMixin

from tools.models.extended_file_manager_model import ModelExtendedManager


class TargetedTransformer(TransformerMixin, ModelExtendedManager):
    """
    Dummy class that allows us to modify only the methods that interest us,
    avoiding redundancy.
    """

    def __init__(self, columns_to_apply=None, models=None, mandatory_attr_only=False, extra_information=None,
                 **model_kwargs):
        self._columns_to_apply = columns_to_apply
        self._array_column_names = []
        ModelExtendedManager.__init__(self, models, mandatory_attr_only, extra_information, **model_kwargs)

    def _get_features(self, x):
        return x.columns if self._columns_to_apply is None else self._columns_to_apply

    def fit(self, x, y=None):
        self._array_column_names = []
        self._models = {}
        features = self._get_features(x)

        for feature in features:
            self._models[feature] = self._individual_fit(x, feature)
            self._set_individual_array_column_name(feature)
        return self

    def _set_individual_array_column_name(self, feature):
        self._array_column_names.append(feature)

    @abstractmethod
    def _individual_fit(self, x, feature):
        pass

    def transform(self, x, y=None):
        features = self._get_features(x)
        transformed_features = [self._individual_transform(x, feature) for feature in features]
        transformed_features = np.concatenate(transformed_features, axis=1)
        return transformed_features

    def _individual_transform(self, x, feature):
        return self._models[feature].transform(x[[feature]])

    def _set_mandatory_attributes_from_models(self):
        self._columns_to_apply = list(self._models.keys())

    def _set_optional_attributes_from_models(self):
        self._array_column_names = []
        for model_name, model in self._models.items():
            self._array_column_names.extend(self._get_array_column_names_from_model(model, model_name))

    @staticmethod
    def _get_array_column_names_from_model(model: Optional = None, model_name: str = ''):
        return [model_name]


class NoTransformer(TargetedTransformer):

    def _individual_fit(self, x, feature):
        return None

    def _individual_transform(self, x, feature):
        return x[feature].values
