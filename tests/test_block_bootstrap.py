"""Tests for BlockBootstrap module"""

import pandas as pd
import numpy as np
from sgr_tools.ScenarioGeneration.BlockBootstrap import BlockBootstrap


def test_index_population_length_overlapping_blocks():
    input_data = pd.DataFrame(np.random.randn(100, 4))
    block_size = 10
    index_population = BlockBootstrap(input_data)._index_population(
        block_size, overlapping_blocks=True
    )
    assert len(index_population) == len(input_data) - block_size + 1


def test_index_population_length_no_overlapping_blocks():
    input_data = pd.DataFrame(np.random.randn(100, 4))
    block_size = 10
    index_population = BlockBootstrap(input_data)._index_population(
        block_size, overlapping_blocks=False
    )
    assert len(index_population) == int(np.floor(len(input_data) / block_size))


def test_scenario_size():
    input_data = pd.DataFrame(np.random.randn(100, 4))
    block_size = 10
    n_blocks = 100
    scenario = BlockBootstrap(input_data)._generate_single_scenario(
        n_blocks=n_blocks, block_size=block_size
    )
    assert len(scenario) == n_blocks * block_size


def test_block_length():
    input_data = pd.DataFrame(np.random.randn(100, 4))
    block_size = 24
    index_population = BlockBootstrap(input_data)._index_population(block_size)
    assert len(index_population[0]) == block_size


def test_months_of_year_subset():
    months_of_year_sets = [
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {10, 11, 12},
    ]
    target_index = pd.Timestamp("2022-06-01")
    expected_result = [4, 5, 6]
    result = BlockBootstrap._months_of_year_subset(months_of_year_sets, target_index)
    assert result == expected_result


def test_days_of_week_subset():
    days_of_week_sets = [
        {0, 1, 2},
        {3, 4},
        {5, 6},
    ]
    target_index = pd.Timestamp("2022-06-01")
    expected_result = [0, 1, 2]
    result = BlockBootstrap._days_of_week_subset(days_of_week_sets, target_index)
    assert result == expected_result


def test_hours_of_day_subset():
    hours_of_day_sets = [
        {0, 1, 2},
        {3, 4},
        {5, 6},
        {7, 8, 9},
        {10, 11, 12},
        {13, 14},
        {15, 16, 17},
        {18, 19},
        {20, 21, 22},
        {23},
    ]
    target_index = pd.Timestamp("2022-06-01 14:30:00")
    expected_result = [13, 14]
    result = BlockBootstrap._hours_of_day_subset(hours_of_day_sets, target_index)
    assert result == expected_result


def test_index_population():
    input_index = pd.date_range(start="1/1/2022", end="1/1/2023", freq="H")[:8760]
    input_data = pd.DataFrame({"value": np.random.randn(8760)}, index=input_index)
    bb = BlockBootstrap(input_data=input_data)

    # Test with no subsets
    index_population = bb._index_population(block_size=24)
    assert len(index_population) == 365
    assert all(len(block) == 24 for block in index_population)

    # Test with months_of_year_subset
    index_population = bb._index_population(
        block_size=24, months_of_year_subset=[1, 2, 3]
    )
    assert len(index_population) == 31 + 28 + 31
    assert all(len(block) == 24 for block in index_population)
    assert all(input_index[block[0]].month in [1, 2, 3] for block in index_population)

    # Test with days_of_week_subset
    index_population = bb._index_population(
        block_size=24, days_of_week_subset=[0, 1, 2]
    )
    assert len(index_population) == 156
    assert all(len(block) == 24 for block in index_population)
    assert all(
        input_index[block[0]].dayofweek in [0, 1, 2] for block in index_population
    )

    # Test with hours_of_day_subset
    index_population = bb._index_population(
        block_size=24, hours_of_day_subset=[0, 1, 2, 3, 4, 5]
    )
    assert len(index_population) == 365
    assert all(len(block) == 24 for block in index_population)
    assert all(
        input_index[block[0]].hour in [0, 1, 2, 3, 4, 5] for block in index_population
    )


def test_seasonality():
    for _ in range(10):
        input_data = pd.DataFrame(
            np.linspace(0, 8759, 8760),
            index=pd.date_range(start="1/1/2050", end="1/1/2051", freq="H")[:8760],
        )

        block_size = np.random.choice([6, 24, 72, 168, 744])
        n_scenarios = np.random.randint(1, 10)
        start_day = np.random.randint(0, 364)
        end_day = np.random.randint(start_day + 1, 365)
        start_idx = start_day * 24
        end_idx = end_day * 24

        months_of_year_sets = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        days_of_week_sets = [[0, 1, 2, 3, 4], [5, 6]]
        target_indices = pd.date_range(start="1/8/2019", end="30/7/2020", freq="H")[
            start_idx:end_idx
        ]
        scenarios = BlockBootstrap(input_data).generate_scenarios(
            n_scenarios=n_scenarios,
            target_indices=target_indices,
            months_of_year_sets=months_of_year_sets,
            days_of_week_sets=days_of_week_sets,
            block_size=block_size,
        )

        for n in range(n_scenarios):
            assert len(scenarios[n]) == len(target_indices)
            i = 0
            while i < len(target_indices):
                val = int(scenarios[n].to_numpy().flatten().tolist()[i])
                month_subset = np.unique(
                    [
                        m_set
                        for m_set in months_of_year_sets
                        if target_indices[i].month in m_set
                    ]
                ).tolist()
                days_subset = np.unique(
                    [
                        d_set
                        for d_set in days_of_week_sets
                        if target_indices[i].dayofweek in d_set
                    ]
                ).tolist()
                assert input_data.index.month[val] in month_subset
                assert input_data.index.dayofweek[val] in days_subset
                i += block_size
