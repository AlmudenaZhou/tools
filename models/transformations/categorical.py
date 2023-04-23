import re
from typing import Optional

import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from tools.models.transformations.base import SklearnTargetedTransformer


class OneHotTargetedTransformer(SklearnTargetedTransformer):

    def __init__(self, columns_to_apply=None, models=None, mandatory_attr_only=False, extra_information=None,
                 **model_kwargs):
        super().__init__(columns_to_apply, models, mandatory_attr_only, extra_information, **model_kwargs)

    def fit(self, x, y=None):
        return super().fit(x)

    def _individual_fit(self, x, feature):
        ohe = OneHotEncoder(sparse=False, **self._model_kwargs).fit(x[[feature]])
        column_names = [f'{feature}__{self._clean_category_name(category)}' for category in ohe.categories_[0]]
        self._array_column_names.extend(column_names)
        return ohe

    @staticmethod
    def _clean_category_name(category):
        category = category.lower()
        category = re.sub('\s+', '_', category)
        return category

    def transform(self, x, y=None):
        return super().transform(x)

    def _individual_transform(self, x, feature):
        return self._models[feature].transform(x[[feature]])

    def _get_array_column_names_from_model(self, model: Optional = None, model_name: str = ''):
        return [f'{model_name}__{self._clean_category_name(category)}' for category in model.categories_[0]]


class OrdinalTargetedTransformer(SklearnTargetedTransformer):

    def __init__(self, ordered_labels=None, columns_to_apply=None, models=None, mandatory_attr_only=False,
                 extra_information=None, **model_kwargs):
        self._ordered_labels = ordered_labels
        super().__init__(columns_to_apply, models, mandatory_attr_only, extra_information, **model_kwargs)

    def fit(self, x, y=None):
        return super().fit(x)

    def _individual_fit(self, x, feature):
        """
        Issue: handle_unknown: 'use_encoded_value' is not properly working and throws an error
        """
        self._array_column_names.append(feature)
        if 'handle_unknown' in self._model_kwargs and self._model_kwargs['handle_unknown'] == 'use_encoded_value':
            return self._fit_model_with_nans(x, feature)
        else:
            return self._fit_model(x, feature)

    def _fit_model(self, x, feature):
        if self._ordered_labels is None:
            raise ValueError('If you want to fit the model, you must specify the _ordered_labels.')

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
        if 'handle_unknown' in self._model_kwargs and self._model_kwargs['handle_unknown'] == 'use_encoded_value':
            return self._transform_data_with_nans(x, feature)
        else:
            return self._transform_data(x, feature)

    def _transform_data(self, x, feature):
        return self._models[feature].transform(x[[feature]])

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
        transf_values[x[feature].notna()] = self._models[feature].transform(x.loc[x[feature].notna(), [feature]])
        return transf_values

    def _set_mandatory_attributes_from_models(self):
        super()._set_mandatory_attributes_from_models()
        self._set_ordered_labels()

    def _set_ordered_labels(self):
        if len(self._models.values()) == 0:
            raise ValueError('_models attribute cannot be empty')

        self._ordered_labels = {}
        for model_name, model in self._models.items():
            self._ordered_labels[model_name] = model.categories[0]
