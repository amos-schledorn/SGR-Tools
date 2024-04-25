import pandas as pd

from ..Models.ForecastModel import ForecastModel
from ...Structures import ScenarioSet


class PeriodicForecast:
    """A class for simulating periodic scenarios using a given model.

    Attributes:
        model (BaseModel): The model to use for forecasting.
        model_is_fit (bool): Indicates whether the model has been fit to the training data.
        _scenarios (dict): A dictionary containing simulated scenarios for each time step.
        _observations (dict): A dictionary containing the observed data for each time step.
        _observations_long (pd.DataFrame): The observed data for the entire simulation period.
    """

    def __init__(self, model: ForecastModel) -> None:
        """Initializes a PeriodicForecast object.

        Args:
            model (ForecastModel): The model to use for forecasting.
        """

        self.model = model
        self.model_is_fit = False

    def simulate_periodic_scenario_sets(
        self,
        observations: pd.DataFrame,
        exog: pd.DataFrame,
        n_train: int,
        n_scen: int,
        t_start: pd.DatetimeIndex,
        t_end: pd.DatetimeIndex,
        forecast_freq: int,
        scenario_len: int,
        datetime_freq: str = "h",
        threads: int = -1,
        **fit_model_kwargs,
    ) -> None:
        """Simulate scenarios n_steps ahead.

        Args:
            observations (pd.DataFrame): The observed data.
            exog (pd.DataFrame): The exogenous variables.
            n_train (int): The number of time steps to use for training the model.
            n_scen (int): The number of scenarios to simulate.
            t_start (pd.DatetimeIndex): The start time of the simulation.
            t_end (pd.DatetimeIndex): The end time of the simulation.
            forecast_freq (int): The frequency of the forecasts.
            scenario_len (int): The length of each scenario.
            datetime_freq (str, optional): The frequency of the time steps. Defaults to "h".
            threads (int, optional): The number of threads to use for parallel processing. Defaults to -1.
            **fit_model_kwargs: Additional keyword arguments to pass to the model's fit method.
        """

        if datetime_freq != "h":
            raise NotImplementedError(
                "Only hourly data are currently supported. Please set datetime_freq='h'."
            )

        if t_start - pd.Timedelta(n_train, unit=datetime_freq) < observations.index[0]:
            raise ValueError(
                f"t_start must be at least {n_train} {datetime_freq} after the first observation."
            )

        # initialise t and ret_val
        t: pd.DatetimeIndex = t_start
        self._scenarios: dict = {}
        self._observations: dict = {}
        self._observations_long = observations.loc[
            pd.date_range(t_start, t_end, freq=datetime_freq), :
        ]
        # while end not reached
        while t < t_end:
            # set training data to t - n_train to t
            t_train = pd.date_range(
                start=t - pd.Timedelta(n_train, unit=datetime_freq),
                periods=n_train,
                freq=datetime_freq,
            )
            # fit model
            self.model.fit(
                y_train=observations.loc[t_train, :],
                exog_train=exog.loc[t_train, :],
                **fit_model_kwargs,
            )
            t_simulation = pd.date_range(
                start=t, periods=scenario_len, freq=datetime_freq
            )
            # simulate scenarios from t to t + scenario_len
            self._scenarios[t] = self.model.simulate_scenarios(
                y0=observations.loc[:t, :],
                exog=exog.loc[t_simulation, :],
                t_range=t_simulation,
                n_scen=n_scen,
                threads=threads,
            )
            # store observations
            self._observations[t] = observations.loc[t_simulation, :]
            # make sure model is not used again on same training data
            self.model_is_fit = False
            # update t to t + forecast_freq
            t += pd.Timedelta(forecast_freq, unit=datetime_freq)

    def get_scenarios(self, t: int | str | pd.Timestamp):
        """Returns the simulated scenarios for a given time step.

        Args:
            t (int | str | pd.Timestamp): The time step.

        Returns:
            pd.DataFrame: The simulated scenarios.
        Raises:
            ValueError: If t is not an int, str or pd.Timestamp.
        """
        if isinstance(t, int):
            return list(self._scenarios.values())[t]
        elif isinstance(t, str):
            # convert t to datetime index
            return self._scenarios[pd.to_datetime(t)]
        elif isinstance(t, pd.Timestamp):
            return self._scenarios[t]
        else:
            raise ValueError("t must be int, str or pd.DatetimeIndex")

    def get_observations(self, t: int | str | pd.DatetimeIndex = None):
        """Returns the observed data for a given time step or the entire simulation period.

        Args:
            t (int | str | pd.DatetimeIndex, optional): The time step. Defaults to None.

        Returns:
            pd.DataFrame: The observed data.
        Raises:
            ValueError: If t is not an int, str or pd.DatetimeIndex.
        """
        if t is None:
            return self._observations_long
        else:
            if isinstance(t, int):
                return list(self._observations.values())[t]
            elif isinstance(t, str):
                # convert t to datetime index
                return self._observations[pd.to_datetime(t)]
            elif isinstance(t, pd.Timestamp):
                return self._observations[t]
            else:
                raise ValueError("t must be int, str or pd.DatetimeIndex")
