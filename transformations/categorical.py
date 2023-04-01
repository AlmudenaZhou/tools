import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from tools.transformations.base import SklearnTargetedTransformer


class OneHotTargetedTransformer(SklearnTargetedTransformer):

    def __init__(self, columns_to_apply, **model_kwargs):
        super().__init__(columns_to_apply, **model_kwargs)

    def fit(self, x, y=None):
        return super().fit(x)

    def _individual_fit(self, x, feature):
        ohe = OneHotEncoder(sparse=False, **self._model_kwargs).fit(x[[feature]])
        column_names = [f'{feature}_{category}' for category in ohe.categories_[0]]
        self._array_column_names.extend(column_names)
        return ohe

    def transform(self, x, y=None):
        return super().transform(x)

    def _individual_transform(self, x, feature):

        return self._encoders[feature].transform(x[[feature]])


class OrdinalTargetedTransformer(SklearnTargetedTransformer):

    def __init__(self, ordered_labels, columns_to_apply=None, **model_kwargs):
        super().__init__(columns_to_apply, **model_kwargs)
        self._ordered_labels = ordered_labels

    def fit(self, x, y=None):
        return super().fit(x)

    def _individual_fit(self, x, feature):
        """
        Issue: handle_unknown: 'use_encoded_value' is not properly working and throws an error
        """
        self._array_column_names.append(feature)
        if self._model_kwargs['handle_unknown'] == 'use_encoded_value':
            return self._fit_model_with_nans(x, feature)
        else:
            return self._fit_model(x, feature)

    def _fit_model(self, x, feature):
        return OrdinalEncoder(categories=[self._ordered_labels[feature]],
                              **self._model_kwargs).fit(x[[feature]])

    def _fit_model_with_nans(self, x, feature):
        x_filtered = x.loc[x[feature].notna()]
        return self._fit_model(x_filtered, feature)

    def transform(self, x, y=None):
        return super().transform(x)

    def _individual_transform(self, x, feature):
        """
        Since the handle unknown is broken in transform, I will bypass it
        """
        if self._model_kwargs['handle_unknown'] == 'use_encoded_value':
            return self._transform_data_with_nans(x, feature)
        else:
            return self._transform_data(x, feature)

    def _transform_data(self, x, feature):
        return self._encoders[feature].transform(x[[feature]])

    def _transform_data_with_nans(self, x, feature):
        """
        unknown_value by default is np.nan as it is said in
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
        """
        if 'unknown_value' in self._model_kwargs.keys():
            unknown_value = self._model_kwargs['unknown_value']
        else:
            unknown_value = np.nan
        transf_values = np.repeat(unknown_value, x.shape[0]).reshape(x.shape[0], 1)
        transf_values[x[feature].notna()] = self._encoders[feature].transform(x.loc[x[feature].notna(), [feature]])
        return transf_values
