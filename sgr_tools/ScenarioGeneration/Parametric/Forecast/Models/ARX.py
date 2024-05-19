import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from .ForecastModel import ForecastModel


class ARX:
    """
    Autoregressive model with exogenous variables.

    This class implements an autoregressive model with exogenous variables (ARX). The model is fit using OLS and can be used to make one-step predictions or simulate scenarios n_steps ahead. The model can be fit with a maximum and minimum value for the dependent variable. The model can also return the noise of the fit.

    Attributes:
        model (None): The fitted model.
        _model_is_fit (bool): Whether the model is fit or not.
        min_val (float): The minimum value allowed for the dependent variable.
        max_val (float): The maximum value allowed for the dependent variable.

    Methods:
        fit(y_train: pd.DataFrame, exog_train: pd.DataFrame, lags: list, intercept: bool=True, max_val: float=None, min_val: float=None) -> None:
            Fits an ARX model using OLS.

        _set_min_val(y_train: pd.Series, min_val: float=None) -> None:
            Sets the minimum value for the target variable.

        _set_max_val(max_val: float=None) -> None:
            Sets the maximum value for the model's output.

        get_endog(y: pd.DataFrame, lags: list = None, exog: pd.DataFrame = None, intercept: bool = True) -> pd.DataFrame:
            Construct design matrix for ARX1 model.

        _clip_value(value) -> pd.Series:
            Clips the value between min_val and max_val.

        predict(endog) -> np.ndarray:
            Predicts the dependent variable.

        get_noise() -> float:
            Returns the noise of the fit.

        _simulate_single_scenario(y0: float, exog: pd.DataFrame, t_range: pd.DatetimeIndex) -> pd.Series:
            Simulate scenario n_steps ahead.

        simulate_scenarios(y0: pd.Series, exog: pd.DataFrame, t_range: pd.DatetimeIndex, n_scen: int) -> pd.DataFrame:
            Simulates multiple scenarios n_steps ahead.
    """

    def __init__(
        self,
    ):
        self.model = None
        self._model_is_fit = False

    def fit(
        self,
        y_train: pd.DataFrame,
        exog_train: pd.DataFrame,
        lags: list,
        intercept: bool = True,
        max_val: float = None,
        min_val: float = None,
    ) -> None:
        """
        Fits an ARX model to the training data.

        Args:
            y_train (pd.DataFrame): The dependent variable training data.
            exog_train (pd.DataFrame): The exogenous variable training data.
            lags (list): The lags to include in the model.
            intercept (bool, optional): Whether to include an intercept. Defaults to True.
            max_val (float, optional): The maximum value allowed for the dependent variable. Defaults to None.
            min_val (float, optional): The minimum value allowed for the dependent variable. Defaults to None.

        Returns:
            None
        """
        # set upper and lower bounds
        self._set_max_val(max_val=max_val)
        self._set_min_val(min_val=min_val, y_train=y_train)

        if lags != [1]:
            raise NotImplementedError("Only ARX1 implemented")
        x_train = ARX.get_x_train(
            y=y_train,
            lags=lags,
            exog=exog_train,
            intercept=intercept,
        )

        mod = sm.regression.linear_model.OLS(y_train.iloc[np.max(lags) :], x_train)
        self.model = mod.fit()
        self.coefs = np.array(self.model.params)
        self.std = np.std(self.model.resid)
        self.lags = lags
        self._model_is_fit = True

    def _set_min_val(self, y_train: pd.Series, min_val: float = None):
        """
        Sets the minimum value for the target variable.

        Args:
            y_train (pd.Series): The target variable in the training data.
            min_val (float, optional): The minimum value to set. Defaults to None.
        """
        self.min_val = (
            (min_val or 0) if y_train.min() >= 0 else -np.inf
        )

    def _set_max_val(self, max_val: float = None):
        """
        Sets the maximum value for the model's output.

        Args:
            max_val (float, optional): The maximum value for the model's output. Defaults to None.
        """
        self.max_val = max_val or np.inf

    @staticmethod
    def get_x_train(
            y: pd.Series,
            lags: list = None,
            exog: np.ndarray = None,
            intercept: bool = True,
        ) -> np.ndarray:
        """
        Returns the training data for the ARX model.

        Args:
            y (pd.Series): The target variable.
            lags (list, optional): A list of lag values to include in the model. Defaults to None.
            exog (np.ndarray, optional): The exogenous variables. Defaults to None.
            intercept (bool, optional): Whether to include an intercept term in the model. Defaults to True.

        Returns:
            np.ndarray: The training data for the ARX model.
        """

        max_lag = np.max(lags)
        all_lags = np.lib.stride_tricks.sliding_window_view(y, (max_lag+1,))
        ret_val = all_lags[:, max_lag - np.array(lags)]
        if intercept:
            ret_val = np.concatenate(
                (ret_val, np.ones(len(ret_val)).reshape(-1, 1)), axis=1
            )
        if exog is not None:
            ret_val = np.concatenate((ret_val, exog.loc[y.index[max_lag:], :]), axis=1)

        return ret_val

    @staticmethod
    def get_x_sim(
        y0: float,
        lags: list = None,
        exog: np.ndarray = None,
        intercept: bool = True,
    ) -> np.ndarray:
        """
        Construct the design matrix for an ARX1 model.

        Args:
            y0 (float): The endogenous variable.
            exog (np.ndarray, optional): The exogenous variables. Defaults to None.
            intercept (bool, optional): Whether or not to include an intercept in the design matrix. Defaults to True.

        Returns:
            np.ndarray: The design matrix.
        """

        max_lag = np.max(lags)
        all_lags = np.lib.stride_tricks.sliding_window_view(np.array(y0).flatten(), (max_lag,))
        ret_val = all_lags[-1, -np.array(lags)]
        if intercept:
            ret_val = np.concatenate(
                (ret_val, [1]))
        if exog is not None:
            ret_val = np.concatenate((ret_val, exog))

        return ret_val

    def _clip_value(self, value: float):
        """
        Clips the given value to the range [min_val, max_val].

        Args:
            value (float): The value to be clipped.

        Returns:
            float: The clipped value.
        """

        return np.clip(value, self.min_val, self.max_val)

    @property
    def model_is_fit(self):
        """
        Returns whether the ARX model has been fit to the data.
        """
        return self._model_is_fit

    @model_is_fit.setter
    def model_is_fit(self, value: bool):
        """
        Sets whether the model has been fit or not.

        Args:
            value (bool): Whether the model has been fit or not.

        Raises:
            ValueError: If value is not a boolean.
        """
        if not isinstance(value, bool):
            raise ValueError("Value must be boolean")
        self._model_is_fit = value

    def predict(self, x_sim: np.ndarray) -> float:
        """
        Predicts the response variable using the fitted ARX model.

        Parameters:
        -----------
        x_sim: np.ndarray
            The endogenous variable used to make the prediction.

        Returns:
        --------
        float
            The predicted value of the response variable.
        """
        return (x_sim * self.coefs).sum()

    def get_noise(self):
        """
        Returns the noise of the ARX model.

        Returns:
        -------
        float
            The noise of the ARX model.
        """
        return np.random.normal(loc=0.0, scale=self.std)
    
    def _simulate_one_step(self, y0: float, exog: np.ndarray, noise: float=None) -> float:
        """Simulate one time step ahead.

        Make 1-step prediction based on previous observation and exogenous variable and add noise.
        
        Args:
            y0 (float): The previous observation.
            exog (np.ndarray): The exogenous variables.
            noise (float, optional): The noise to add to the prediction. Defaults to None. If None, noise is generated.
        
        Returns:
            float: The predicted value for the next time step.
        """
        if self.model_is_fit:
            x_sim = ARX.get_x_sim(y0=y0, exog=exog, lags=self.lags)
            pred = self.predict(x_sim=x_sim)
            noise = noise or self.get_noise()
        else:
            raise ValueError("Model has not been fit yet")
        
        return self._clip_value(pred + noise)

    def _simulate_single_scenario(
        self,
        y0: float,
        exog: np.ndarray,
        length: int,   
    ) -> pd.Series:
        """
        Simulate a single scenario n_steps ahead.

        This method iteratively makes 1-step predictions and then bases the next prediction on the previous prediction
        (assuming an ARX1 model).

        Parameters:
        -----------
        y0 : float
            The initial value of the dependent variable.
        exog : np.ndarray
            The exogenous variables used in the simulation.
        length : int
            The scenario length.

        Returns:
        --------
        np.ndarray
            A numpy array containing the simulated scenario.
        """

        ret_val = np.zeros(length)
        for i in range(length):
            if i > 0:
                y0 = np.hstack((y0, ret_val[i-1]))
            ret_val[i] = self._simulate_one_step(y0=y0, exog=exog[i] if exog is not None else None)

        return ret_val

    def simulate_scenarios(
        self,
        y0: float | np.ndarray,
        exog: pd.DataFrame,
        t_range: pd.DatetimeIndex,
        n_scen: int,
        threads=-1,
    ) -> pd.DataFrame:
        """
        Simulate multiple scenarios for the ARX model.

        Args:
            y0 (float | np.ndarray): The initial values for the time series.
            exog (pd.DataFrame): The exogenous variables for the model.
            t_range (pd.DatetimeIndex): The time range for the simulation.
            n_scen (int): The number of scenarios to simulate.

        Returns:
            pd.DataFrame: A DataFrame containing the simulated scenarios.
        """

        if exog is not None:
            exog = exog.loc[t_range, :].to_numpy()
        scenario_length = len(t_range)

        with Parallel(n_jobs=threads) as parallel:
            scenarios = parallel(
                delayed(self._simulate_single_scenario)(
                    y0,
                    exog,
                    scenario_length,
                )
                for _ in range(n_scen)
            )


        return pd.DataFrame({f"s{i}": scenarios[i] for i in range(n_scen)}, index=t_range)
