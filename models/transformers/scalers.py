from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tools.models.transformers.base import TargetedTransformer


class StandardScalerTransformer(TargetedTransformer):

    def _individual_fit(self, x, feature):
        return StandardScaler(**self._model_kwargs).fit(x[[feature]])


class MinMaxScalerTransformer(TargetedTransformer):

    def _individual_fit(self, x, feature):
        return MinMaxScaler(**self._model_kwargs).fit(x[[feature]])
