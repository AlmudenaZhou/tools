from abc import ABC


class AbstractModel(ABC):

    def fit(self):
        return self

    def _individual_fit(self):
        pass

    def transform(self):
        pass

    def _individual_transform(self):
        pass
