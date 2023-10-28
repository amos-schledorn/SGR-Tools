import numpy as np
import pandas as pd
from typing import List


class BlockBootstrap:
    """
    Generates time series scenarios using block bootstrap.

    """

    def __init__(self, input_data: pd.DataFrame) -> None:
        """
        Initialize a new instance of the BlockBootstrap class.

        Parameters
        ----------
        input_data : pandas.DataFrame
        The input time series data to be resampled.
        """
        self.input_data = input_data.copy()
        self.input_index = self.input_data.index

    def generate_scenarios(
        self,
        n_scenarios: int,
        block_size: int,
        n_blocks: int = None,
        target_indices: pd.DatetimeIndex = None,
        months_of_year_sets: list = None,
        days_of_week_sets: list = None,
        hours_of_day_sets: list = None,
        overlapping_blocks: bool = False,
    ) -> list:
        """
        Generates n_scenarios scenarios of length `n_blocks * block_size` from the input data.

        Parameters
        ----------
        n_scenarios : int
            The number of scenarios to generate.
        block_size : int
            The size of each block in the population.
        n_blocks : int, optional
            The number of blocks to include in each scenario. If None, the number of blocks is determined by the length of the input data.
        target_indices : pandas.DatetimeIndex, optional
            A DatetimeIndex of target indices to include in each scenario. If None, no target indices are included.
        months_of_year_sets : list of set, optional
            A list of sets of months of the year to include in the population.
        days_of_week_sets : list of set, optional
            A list of sets of days of the week to include in the population.
        hours_of_day_sets : list of set, optional
            A list of sets of hours of the day to include in the population.
        overlapping_blocks : bool, default False
            Whether to allow overlapping blocks in the population.

        Returns
        -------
        scenarios : list of pandas.DataFrame
            A list of DataFrames containing the generated scenarios.
        """

        scenarios = []
        for _ in range(n_scenarios):
            scenarios.append(
                self._generate_single_scenario(
                    n_blocks=n_blocks,
                    block_size=block_size,
                    target_indices=target_indices,
                    months_of_year_sets=months_of_year_sets,
                    days_of_week_sets=days_of_week_sets,
                    hours_of_day_sets=hours_of_day_sets,
                    overlapping_blocks=overlapping_blocks,
                )
            )

        return scenarios

    def _validate_input(self, n_blocks: int, target_indices: pd.DatetimeIndex) -> None:
        """
        Validates the input parameters for the generate_scenarios method.

        Parameters
        ----------
        n_blocks : int
            The number of blocks to include in each scenario. If None, the number of blocks is determined by the length of the input data.
        target_indices : pandas.DatetimeIndex
            A DatetimeIndex of target indices to include in each scenario. If None, no target indices are included.

        Raises
        ------
        ValueError
            If both n_blocks and target_indices are None, or if both are not None.
            If target_indices is supplied but input_data index is not a DatetimeIndex.
            If target_indices is supplied but is not a DatetimeIndex.
        """
        if n_blocks is None and target_indices is None:
            raise ValueError("Either n_blocks or target_indices must be not None.")
        if n_blocks is not None and target_indices is not None:
            raise ValueError("Only one of n_blocks or target_indices can be not None.")
        if target_indices is not None and not isinstance(
            self.input_data.index, pd.DatetimeIndex
        ):
            raise ValueError(
                "If target_indices is supplied, input_data index must be a DatetimeIndex."
            )
        if target_indices is not None and not isinstance(
            target_indices, pd.DatetimeIndex
        ):
            raise ValueError(
                "If target_indices is supplied, it must be a DatetimeIndex."
            )

    def _generate_single_scenario(
        self,
        block_size: int,
        n_blocks: int = None,
        target_indices: pd.DatetimeIndex = None,
        months_of_year_sets: list = None,
        days_of_week_sets: list = None,
        hours_of_day_sets: list = None,
        overlapping_blocks: bool = False,
    ) -> pd.DataFrame:
        """
        Generates a scenario of length `n_blocks * block_size` from the input data.

        Parameters
        ----------
        block_size : int
            The size of each block in the scenario.
        n_blocks : int, optional
            The number of blocks to include in the scenario. Defaults to None. If not supplied, target_indices is used.
        target_indices : pandas.DatetimeIndex, optional
            A DatetimeIndex of target indices to include in the scenario. Defaults to None. If not supplied, n_blocks is used
        months_of_year_sets : list of set, optional
            A list of sets of months of the year to include in the scenario.
        days_of_week_sets : list of set, optional
            A list of sets of days of the week to include in the scenario.
        hours_of_day_sets : list of set, optional
            A list of sets of hours of the day to include in the scenario.
        overlapping_blocks : bool, default False
            Whether to allow overlapping blocks in the scenario.

        Returns
        -------
        scenario : pandas.DataFrame
            The generated scenario.
        """

        scenario_indices = []
        while True:
            target_index = (
                target_indices[len(scenario_indices)]
                if target_indices is not None
                else None
            )

            index_population = self._index_population(
                block_size=block_size,
                months_of_year_subset=BlockBootstrap._months_of_year_subset(
                    months_of_year_sets=months_of_year_sets, target_index=target_index
                ),
                days_of_week_subset=BlockBootstrap._days_of_week_subset(
                    days_of_week_sets=days_of_week_sets, target_index=target_index
                ),
                hours_of_day_subset=BlockBootstrap._hours_of_day_subset(
                    hours_of_day_sets=hours_of_day_sets, target_index=target_index
                ),
                overlapping_blocks=overlapping_blocks,
            )

            block_index = np.random.choice([i for i in range(len(index_population))])
            scenario_indices += index_population[block_index]

            if (n_blocks and len(scenario_indices) >= n_blocks * block_size) or (
                target_indices is not None
                and len(scenario_indices) >= len(target_indices)
            ):
                break

        scenario = self.input_data.iloc[scenario_indices, :]

        scenario = BlockBootstrap._apply_target_indices(scenario, target_indices)

        return scenario

    def _index_population(
        self,
        block_size: int,
        overlapping_blocks: bool = False,
        months_of_year_subset: list = None,
        days_of_week_subset: list = None,
        hours_of_day_subset: list = None,
    ) -> list:
        """
        Generates a population of block indices from the input data.

        Parameters
        ----------
        block_size : int
            The size of each block in the population.
        overlapping_blocks : bool, default False
            Whether to allow overlapping blocks in the population.
        months_of_year_subset : list of set, optional
            A list of sets of months of the year to include in the population.
        days_of_week_subset : list of set, optional
            A list of sets of days of the week to include in the population.
        hours_of_day_subset : list of set, optional
            A list of sets of hours of the day to include in the population.

        Returns
        -------
        index_population : list of list of int
            The population of block indices.
        """

        # convert input index to numpy array for faster indexing
        # input_index = np.array(self.input_index)
        input_index = self.input_index

        # Create boolean masks for each subset condition
        months_mask = np.full(len(input_index), True)
        days_mask = np.full(len(input_index), True)
        hours_mask = np.full(len(input_index), True)

        if months_of_year_subset:
            months_mask = np.isin(input_index.month, months_of_year_subset)
        if days_of_week_subset:
            days_mask = np.isin(input_index.dayofweek, days_of_week_subset)
        if hours_of_day_subset:
            hours_mask = np.isin(input_index.hour, hours_of_day_subset)

        # Combine the masks into a single mask
        mask = months_mask & days_mask & hours_mask

        index_population = []
        i = 0
        while i + block_size <= len(input_index):
            if mask[i]:
                index_population.append([j for j in range(i, i + block_size)])

                if overlapping_blocks:
                    i += 1
                else:
                    i += block_size
            else:
                if hours_of_day_subset:
                    i += 1
                else:
                    i += 24

        return index_population

    @staticmethod
    def _months_of_year_subset(
        months_of_year_sets: list, target_index: pd.Timestamp
    ) -> list:
        """
        Returns the subset of month_of_year_sets that contains the target_index.

        Parameters
        ----------
        months_of_year_sets : list of set or None
            A list of sets of months of the year to search for the target index.
        target_index : pandas.Timestamp
            The target index to search for in the month_of_year_sets.

        Returns
        -------
        subset : list of int or None
            The subset of month_of_year_sets that contains the target_index, or None if months_of_year_sets is None.
        """
        return (
            np.unique(
                [
                    i
                    for subset in months_of_year_sets
                    for i in subset
                    if target_index.month in subset
                ]
            ).tolist()
            if months_of_year_sets is not None
            else None
        )

    @staticmethod
    def _days_of_week_subset(
        days_of_week_sets: list, target_index: pd.Timestamp
    ) -> list:
        """
        Returns the subset of days_of_week_sets that contains the target_index.

        Parameters
        ----------
        days_of_week_sets : list of set or None
            A list of sets of days of the week to search for the target index.
        target_index : pandas.Timestamp
            The target index to search for in the days_of_week_sets.

        Returns
        -------
        subset : list of int or None
            The subset of days_of_week_sets that contains the target_index, or None if days_of_week_sets is None.
        """
        return (
            np.unique(
                [
                    i
                    for subset in days_of_week_sets
                    for i in subset
                    if target_index.dayofweek in subset
                ]
            ).tolist()
            if days_of_week_sets is not None
            else None
        )

    @staticmethod
    def _hours_of_day_subset(
        hours_of_day_sets: list, target_index: pd.Timestamp
    ) -> list:
        """
        Returns the subset of hours_of_day_sets that contains the target_index.

        Parameters
        ----------
        hours_of_day_sets : list of set or None
            A list of sets of hours of the day to search for the target index.
        target_index : pandas.Timestamp
            The target index to search for in the hours_of_day_sets.

        Returns
        -------
        subset : list of int or None
            The subset of hours_of_day_sets that contains the target_index, or None if hours_of_day_sets is None.
        """
        return (
            np.unique(
                [
                    i
                    for subset in hours_of_day_sets
                    for i in subset
                    if target_index.hour in subset
                ]
            ).tolist()
            if hours_of_day_sets is not None
            else None
        )

    @staticmethod
    def _apply_target_indices(
        scenario: pd.DataFrame, target_indices: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Applies target indices to the scenario DataFrame.

        Parameters
        ----------
        scenario : pandas.DataFrame
            The scenario DataFrame to apply the target indices to.
        target_indices : pandas.DatetimeIndex
            The target indices to apply to the scenario DataFrame.

        Returns
        -------
        scenario : pandas.DataFrame
            The scenario DataFrame with the target indices applied.
        """
        if target_indices is not None:
            scenario = scenario.iloc[
                : len(target_indices), :
            ]  # in case block_length is not a multiple of len(target_indices)
            scenario.index = target_indices

        return scenario
