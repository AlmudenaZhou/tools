from abc import abstractmethod, ABC

import numpy as np
from sklearn.base import TransformerMixin


class TargetedTransformer(TransformerMixin):
    """
    Dummy class that allows us to modify only the methods that interest us,
    avoiding redundancy.
    """

    def __init__(self, columns_to_apply=None, **model_kwargs):
        self._columns_to_apply = columns_to_apply
        self._model_kwargs = model_kwargs

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
        self._encoders = {}
        self._array_columns = []

    def fit(self, x, y=None):
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
