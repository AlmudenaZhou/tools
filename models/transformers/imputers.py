from __future__ import annotations

from abc import abstractmethod
from typing import Union

from sklearn.impute import SimpleImputer, KNNImputer

from tools.file_manager_workflows.extended_file_manager_model import ModelExtendedManager


class GenericImputerTransformer(ModelExtendedManager):

    single_parameter_type: type

    def __init__(self, parameters: Union[single_parameter_type, list] = None, order_labels_by_num_nans=False,
                 models=None, mandatory_attr_only=False, extra_information=None, **model_kwargs):
        self._model_kwargs = model_kwargs
        super().__init__(models, mandatory_attr_only, extra_information)
        self.check_parameters(parameters)
        self._parameters = parameters
        self._order_labels_by_num_nans = order_labels_by_num_nans

    def check_parameters(self, parameters):
        if self._models is None and not (isinstance(parameters, (self.single_parameter_type, list))):
            raise TypeError('strategies must be an integer or a list')

    @abstractmethod
    def _set_mandatory_attributes_from_models(self):
        pass

    def _set_optional_attributes_from_models(self):
        pass

    def fit(self, x, y=None, model_name=''):

        if isinstance(self._parameters, self.single_parameter_type):
            self._models[f'{model_name}{self._parameters}'] = self._individual_fit(x, self._parameters)
        else:
            for parameter in self._parameters:
                self._models[f'{model_name}{parameter}'] = self._individual_fit(x, parameter)
        return self

    @abstractmethod
    def _individual_fit(self, x, parameter):
        pass

    def transform(self, x, y=None):
        if isinstance(self._parameters, self.single_parameter_type):
            return list(self._models.values())[0].transform(x)
        return {model_name: model.transform(x) for model_name, model in self._models.items()}


class SimpleImputerTransformer(GenericImputerTransformer):

    single_parameter_type = str

    def _individual_fit(self, x, parameter):
        return SimpleImputer(strategy=parameter, **self._model_kwargs).fit(x)

    def _set_mandatory_attributes_from_models(self):
        self._parameters = [simple_imputer.strategy for simple_imputer in self._models.values()]


class KNNImputerTransformer(GenericImputerTransformer):

    single_parameter_type = int

    def _individual_fit(self, x, parameter):
        return KNNImputer(n_neighbors=parameter, **self._model_kwargs).fit(x)

    def _set_mandatory_attributes_from_models(self):
        self._parameters = [knn.n_neighbors for knn in self._models.values()]
