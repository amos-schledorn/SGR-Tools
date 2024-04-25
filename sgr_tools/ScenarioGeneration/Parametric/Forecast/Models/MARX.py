import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from ...Structures import ScenarioSet
from . import ForecastModel, ARX


class MARX(ForecastModel):
    """Multi-variate ARX model"""

    def __init__(self) -> None:
        """
        Initializes a new instance of the MARX class.
        """
        self.arx_models = None
        self.min_val = None
        self.max_val = None
        self.cov = None

    def fit(
        self,
        y_train: pd.DataFrame,
        exog_train: pd.DataFrame,
        lags: list,
        intercept: bool = True,
    ) -> None:
        """
        Fits the MARX model to the training data.

        Args:
            y_train (pd.DataFrame): The dependent variable training data.
            exog_train (pd.DataFrame): The exogenous variable training data.
            lags (list): A list of lags to include in the model.
            intercept (bool, optional): Whether or not to include an intercept term. Defaults to True.
        """
        self.arx_models = {col: ARX() for col in y_train.columns}
        for col, model in self.arx_models.items():
            model.fit(
                y_train=y_train[col],
                lags=lags,
                exog_train=exog_train,
                intercept=intercept,
            )

        resid = np.row_stack([val.model.resid for val in self.arx_models.values()])
        self.cov = np.cov(resid).reshape(len(self.arx_models), len(self.arx_models))

        self.min_val = np.array([val.min_val for val in self.arx_models.values()])
        self.max_val = np.array([val.max_val for val in self.arx_models.values()])

    @property
    def model_is_fit(self) -> bool:
        """
        Gets a value indicating whether or not the MARX model is fit.

        Returns:
            bool: True if the model is fit; otherwise, False.
        """
        if self.arx_models is None:
            return False
        else:
            return all([val.model_is_fit for val in self.arx_models.values()])

    @model_is_fit.setter
    def model_is_fit(self, value):
        """
        Sets a value indicating whether or not the MARX model is fit.

        Args:
            value: The value to set.
        """
        for model in self.arx_models.values():
            model.model_is_fit = value

    def _get_noise(self):
        """
        Gets the noise for the MARX model.

        Returns:
            np.ndarray: An array of noise values for the MARX model.
        """
        return np.random.multivariate_normal(np.zeros(len(self.arx_models)), self.cov)

    def _simulate_single_scenario(
        self,
        y0: np.ndarray,
        exog: np.ndarray,
        length: int,
    ) -> pd.Series:
        """
        Simulates a single scenario for the MARX model.

        Args:
            y0 (pd.Series): The initial dependent variable values.
            exog (pd.DataFrame): The exogenous variable data.
            t_range (pd.DatetimeIndex): The time range to simulate.

        Returns:
            pd.Series: A Series of simulated dependent variable values for the MARX model.
        """
        ret_val = []
        for i in range(length):
            if i > 0:
                y0 = np.hstack((y0, np.array(ret_val[i-1]).reshape(-1,1)))
            ret_val.append(self._simulate_one_step(y0=y0, exog=exog[i]))
            

        return np.array(ret_val)

    def _simulate_one_step(self, y0: np.ndarray, exog: np.ndarray) -> pd.Series:

        noise = self._get_noise()
        return [val._simulate_one_step(y0=y0[key], exog=exog, noise=noise[key]) for key, val in enumerate(self.arx_models.values())]


    
    def _clip_value(self, value):
        """
        Clips the given value to the minimum and maximum values for the MARX model.

        Args:
            value: The value to clip.

        Returns:
            pd.Series: A Series of clipped values for the MARX model.
        """
        return np.clip(value, self.min_val, self.max_val)

    def simulate_scenarios(
        self, y0: pd.Series, exog: pd.DataFrame, t_range: pd.DatetimeIndex, n_scen: int, threads=-1
    ) -> ScenarioSet:
        """
        Simulates multiple scenarios for the MARX model.

        Args:
            y0 (pd.Series): The initial dependent variable values.
            exog (pd.DataFrame): The exogenous variable data.
            t_range (pd.DatetimeIndex): The time range to simulate.
            n_scen (int): The number of scenarios to simulate.

        Returns:
            dict[pd.DataFrame]: A dictionary of DataFrames containing simulated dependent variable values for the MARX model.
        """

        exog = exog.loc[t_range, :].to_numpy()
        y0 = np.array([y0[key] for key in self.arx_models.keys()])
        if y0.shape[0] != len(self.arx_models):
            raise ValueError(
                f"Length of y0 ({len(y0)}) does not match number of models ({len(self.arx_models)})"
            )
        scenario_length = len(t_range)

        with Parallel(n_jobs=threads) as parallel:
            scenarios = np.array(parallel(delayed(
                self._simulate_single_scenario)(y0, exog, scenario_length) for _ in range(n_scen)))
        
        # ensure shape
        if scenarios.shape[0] != n_scen:
            raise ValueError(
                f"Number of scenarios ({scenarios.shape[0]}) does not match n_scen ({n_scen})"
            )
        if scenarios.shape[1] != scenario_length:
            raise ValueError(
                f"Scenario length ({scenarios.shape[1]}) does not match length of t_range ({scenario_length})")
        if scenarios.shape[2] != len(self.arx_models):
            raise ValueError(
                f"Number of models ({scenarios.shape[2]}) does not match number of models ({len(self.arx_models)})")
        
        

        return ScenarioSet({
            key: pd.DataFrame(
                {f"s{s}": scenarios[s,:,i] for s in range(n_scen)}
            )
            for i, key in enumerate(self.arx_models.keys())
        })
