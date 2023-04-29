from typing import Union

from sklearn.impute import KNNImputer

from tools.models.extended_file_manager_model import ModelExtendedManager


class KNNImputerAdapter(ModelExtendedManager):

    def __init__(self, n_neighbors: Union[int, list], order_labels_by_num_nans=False):
        self.check_n_neighbours(n_neighbors)
        self._n_neighbors = n_neighbors
        self._order_labels_by_num_nans = order_labels_by_num_nans
        super().__init__()

    @staticmethod
    def check_n_neighbours(n_neighbors):
        if not (isinstance(n_neighbors, (int, list))):
            raise TypeError('n_neighbors must be an integer or a list')

    def _set_mandatory_attributes_from_models(self):
        pass

    def _set_optional_attributes_from_models(self):
        pass

    def fit(self, x, y=None, model_name=''):

        if isinstance(self._n_neighbors, int):
            self._models[f'{model_name}{self._n_neighbors}'] = self._individual_fit(x, self._n_neighbors)
        else:
            for n_neighbors in self._n_neighbors:
                self._models[f'{model_name}{n_neighbors}'] = self._individual_fit(x, n_neighbors)
        return self

    @staticmethod
    def _individual_fit(x, n_neighbours):
        return KNNImputer(n_neighbors=n_neighbours).fit(x)

    def transform(self, x, y=None):
        if isinstance(self._n_neighbors, int):
            return list(self._models.values())[0].transform(x)
        return {model_name: model.transform(x) for model_name, model in self._models.items()}
