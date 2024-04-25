
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class ForecastModel(ABC):
    """Base class for forecasting models.
    """

    @abstractmethod
    def fit(self) -> None:
        """Fit model to training data."""
        pass

    @abstractmethod
    def simulate_scenarios(self, y0: pd.Series | float, exog:pd.DataFrame, t_range:pd.DatetimeIndex, n_scen: int):
        """Simulate scenarios n_steps ahead."""
        pass
    
    @abstractmethod
    def _get_noise(self):
        """Generate noise."""
        pass

    @property
    @abstractmethod
    def model_is_fit(self) -> bool:
        """Check if model has been fit."""
        pass

    @model_is_fit.setter
    @abstractmethod
    def model_is_fit(self, value: bool) -> bool:
        pass

    @abstractmethod
    def _simulate_one_step(self, y0: np.ndarray, exog: pd.DataFrame, noise=None) -> pd.Series:
        """Simulate one time step ahead.

        Make 1-step prediction based on previous observation and exogenous variable and add noise.
        """




