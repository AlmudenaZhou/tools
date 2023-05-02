import logging
from abc import ABC, abstractmethod


class ModelExtendedManager(ABC):

    def __init__(self, models=None, mandatory_attr_only=False, extra_information=None, **model_kwargs):
        self._models = None
        self._start_from_model(models, mandatory_attr_only)
        self._model_kwargs = model_kwargs
        self.extra_information = extra_information

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
