"""Tests for ScenarioTree module."""

import numpy as np
from sklearn_extra.cluster import KMedoids
from sgr_tools.ScenarioReduction import ScenarioTree


def test_medoids_2_stage():
    np.random.seed(0)
    n_tests = 10
    for _ in range(n_tests):
        # generate sample data for testing
        n_scenarios_generated = 100
        n_timesteps = 24
        sample = np.random.normal(size=(n_scenarios_generated, n_timesteps))
        n_scenarios_reduced = 10

        clustering = KMedoids(
            metric="euclidean", n_clusters=n_scenarios_reduced, method="pam"
        ).fit(sample)
        expected_result = clustering.cluster_centers_

        # create scenario tree
        tree = ScenarioTree(
            data=sample, split_idx=[0], split_branches=[n_scenarios_reduced]
        )
        result = tree.get_scenario_data()

        assert np.allclose(result, expected_result)


def test_probabilities_2_stage():
    np.random.seed(1)
    n_tests = 10
    for _ in range(n_tests):
        # generate sample data for testing
        n_scenarios_generated = 100
        n_timesteps = 24
        sample = np.random.normal(size=(n_scenarios_generated, n_timesteps))
        n_scenarios_reduced = 10

        clustering = KMedoids(
            metric="euclidean", n_clusters=n_scenarios_reduced, method="pam"
        ).fit(sample)
        expected_result = np.array(
            [
                np.sum([i == n for i in clustering.labels_.tolist()])
                for n in range(n_scenarios_reduced)
            ]
        ) / len(sample)

        # create scenario tree
        tree = ScenarioTree(
            data=sample, split_idx=[0], split_branches=[n_scenarios_reduced]
        )
        result = tree.get_scenario_probabilities()

        assert np.allclose(result, expected_result)


def test_clustering_3_stage():
    np.random.seed(1494)
    n_tests = 20
    for _ in range(n_tests):
        # generate sample data for testing
        n_scenarios_generated = 2000
        n_timesteps = np.random.randint(10, 30)
        sample = _random_walk(
            np.random.rand() ** 2,
            np.random.rand() ** 2,
            n_timesteps,
            n_scenarios_generated,
        )

        split_idx = np.empty(2, int)
        split_idx[0] = 0
        split_idx[1] = np.random.randint(1, n_timesteps - 1)

        split_branches = np.empty(2, int)
        split_branches[0] = np.random.randint(2, 10)
        split_branches[1] = np.random.randint(2, 10)

        expected_scenarios, expected_probabilities = _3_stage_clustering_manual(
            sample, split_idx=split_idx, split_branches=split_branches
        )

        # create scenario tree
        tree = ScenarioTree(
            data=sample, split_idx=split_idx, split_branches=split_branches
        )
        scenarios = tree.get_scenario_data()
        probabilities = tree.get_scenario_probabilities()

        assert np.allclose(scenarios, expected_scenarios)
        assert np.allclose(probabilities, expected_probabilities)


def _random_walk(mean, stdev, n_timesteps, n_samples):
    return np.array(
        [
            np.cumsum(np.random.normal(mean, stdev, n_timesteps))
            for _ in range(n_samples)
        ]
    )


def _3_stage_clustering_manual(sample, split_idx, split_branches):
    n_scenarios = int(np.prod(split_branches))
    scenario_probabilities = np.empty(n_scenarios)
    reduced_scenarios = np.empty((n_scenarios, len(sample[0])))

    clustering = KMedoids(
        metric="euclidean",
        n_clusters=split_branches[0],
        method="pam",
        random_state=1494,
    ).fit(sample)
    medoids_stage1 = clustering.cluster_centers_

    to_cluster = [
        np.array(
            [
                sample[i, split_idx[1] :]
                for i in range(sample.shape[0])
                if clustering.labels_[i] == n
            ]
        )
        for n in range(split_branches[0])
    ]

    for idx, val in enumerate(to_cluster):
        clustering = KMedoids(
            metric="euclidean",
            n_clusters=split_branches[1],
            method="pam",
            random_state=1494,
        ).fit(val)

        reduced_scenarios[
            idx * split_branches[1] : (idx + 1) * split_branches[1], : split_idx[1]
        ] = np.tile(
            medoids_stage1[idx][: split_idx[1]], (split_branches[1], 1)
        ).tolist()
        reduced_scenarios[
            idx * split_branches[1] : (idx + 1) * split_branches[1], split_idx[1] :
        ] = clustering.cluster_centers_

        scenario_probabilities[
            idx * split_branches[1] : (idx + 1) * split_branches[1]
        ] = np.array(
            [
                np.sum([i == n for i in clustering.labels_.tolist()])
                for n in range(split_branches[1])
            ]
        ) / len(
            sample
        )

    return (reduced_scenarios, scenario_probabilities)
