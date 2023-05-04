from __future__ import annotations

from abc import abstractmethod
from typing import Union

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

from tools.file_manager_workflows.extended_file_manager_model import ModelExtendedManager
from tools.file_manager_workflows.model_file_manager_workflows import ManagerWorkflow
from tools.misc import pascal_to_snake_case


class GenericImputerTransformer(ModelExtendedManager):

    single_parameter_type: type
    parameter_name: str

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

    def _set_mandatory_attributes_from_models(self):
        self._parameters = [getattr(imputer, self.parameter_name) for imputer in self._models.values()]

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
    parameter_name = 'strategy'

    def _individual_fit(self, x, parameter):
        return SimpleImputer(strategy=parameter, **self._model_kwargs).fit(x)


class KNNImputerTransformer(GenericImputerTransformer):

    single_parameter_type = int
    parameter_name = 'n_neighbors'

    def _individual_fit(self, x, parameter):
        return KNNImputer(n_neighbors=parameter, **self._model_kwargs).fit(x)


class IterativeImputerTransformer(GenericImputerTransformer):

    parameter_name = 'estimator'

    def check_parameters(self, parameters):
        pass

    @staticmethod
    def _get_paramater_name(parameter):
        model_path = ManagerWorkflow.get_model_class_path_from_model(parameter, is_model_already_a_class=True)
        model_name = model_path.split('.')[-1]
        parameter_name = pascal_to_snake_case(model_name)
        return parameter_name

    def fit(self, x, y=None, model_name=''):

        if isinstance(self._parameters, list):
            for parameter in self._parameters:
                parameter_name = self._get_paramater_name(parameter)
                self._models[f'{model_name}{parameter_name}'] = self._individual_fit(x, parameter)
        else:
            parameter_name = self._get_paramater_name(self._parameters)
            self._models[f'{model_name}{parameter_name}'] = self._individual_fit(x, self._parameters)

        return self

    def _individual_fit(self, x, parameter):
        """
        BayesianRidge and ExtraTreeRegressor recommended
        :param x:
        :param parameter:
        :return:
        """
        return IterativeImputer(estimator=parameter(), **self._model_kwargs).fit(x)
