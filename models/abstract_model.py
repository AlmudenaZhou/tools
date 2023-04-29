from abc import ABC, abstractmethod
from typing import Protocol, Optional, Union

import numpy as np
import pandas as pd


class GenericTransformer(Protocol):

    @abstractmethod
    def fit(self):
        return self

    @abstractmethod
    def transform(self):
        pass


class AbstractTransformer(ABC):

    @abstractmethod
    def fit(self, x: pd.DataFrame, y: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None):
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
