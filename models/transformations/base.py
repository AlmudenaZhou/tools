import logging
from abc import abstractmethod, ABC
from typing import Optional

import numpy as np
from sklearn.base import TransformerMixin


class ModelExtendedManager(ABC):

    def __init__(self, models=None, mandatory_attr_only=False, **model_kwargs):
        self._models = None
        self._start_from_model(models, mandatory_attr_only)
        self._model_kwargs = model_kwargs

    def _start_from_model(self, models, mandatory_attr_only=False):
        self._models = {}
        if models is not None:
            self._models = models
            self._set_attributes_from_models(mandatory_attr_only)

    def _set_attributes_from_models(self, mandatory_only):
        if len(self._models.values()) == 0:
            logging.warning('Models are empty')
            return

        self._set_mandatory_attributes_from_models()
        if not mandatory_only:
            self._set_optional_attributes_from_models()

    @abstractmethod
    def _set_mandatory_attributes_from_models(self):
        pass

    @abstractmethod
    def _set_optional_attributes_from_models(self):
        pass


class TargetedTransformer(TransformerMixin, ModelExtendedManager):
    """
    Dummy class that allows us to modify only the methods that interest us,
    avoiding redundancy.
    """

    def __init__(self, columns_to_apply=None, models=None, mandatory_attr_only=False, **model_kwargs):
        self._columns_to_apply = columns_to_apply
        self._array_column_names = []
        ModelExtendedManager.__init__(self, models, mandatory_attr_only, **model_kwargs)

    def _get_features(self, x):
        return x.columns if self._columns_to_apply is None else self._columns_to_apply

    def fit(self, x, y=None):
        return self

    @abstractmethod
    def transform(self, x, y=None):
        pass

    def _set_mandatory_attributes_from_models(self):
        self._columns_to_apply = list(self._models.keys())

    def _set_optional_attributes_from_models(self):
        self._array_column_names = []
        for model_name, model in self._models.items():
            self._array_column_names.extend(self._get_array_column_names_from_model(model, model_name))

    @staticmethod
    def _get_array_column_names_from_model(model: Optional = None, model_name: str = ''):
        return [model_name]


class SklearnTargetedTransformer(TargetedTransformer, ABC):

    def __init__(self, columns_to_apply=None, models=None, **model_kwargs):
        super().__init__(columns_to_apply, models, **model_kwargs)

    def fit(self, x, y=None):
        self._array_column_names = []
        self._models = {}
        features = self._get_features(x)

        for feature in features:
            self._models[feature] = self._individual_fit(x, feature)
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
