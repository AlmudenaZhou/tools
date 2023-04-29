from abc import ABC, abstractmethod
from typing import Protocol, Optional, Union, TypeVar

import numpy as np
import pandas as pd


Model = TypeVar("Model", bound="AbstractTransformer")
GenericModel = TypeVar("GenericModel", bound="GenericTransformer")


class GenericTransformer(Protocol):

    def fit(self, x: pd.DataFrame, y: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None) -> GenericModel: ...

    def transform(self, x: pd.DataFrame, y: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None) -> np.ndarray:
        ...


class AbstractTransformer(ABC):

    @abstractmethod
    def fit(self, x: pd.DataFrame, y: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None) -> Model:
        return self

    @abstractmethod
    def _individual_fit(self, x: pd.DataFrame, **kwargs) -> GenericTransformer:
        pass

    @abstractmethod
    def transform(self, x: pd.DataFrame, y: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None) -> np.ndarray:
        pass

    @abstractmethod
    def _individual_transform(self, x: pd.DataFrame, **kwargs) -> np.ndarray:
        pass
