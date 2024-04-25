import numpy as np
import pandas as pd
import xarray as xr

from sklearn_extra.cluster import KMedoids


class ScenarioSet:
    """
    A class representing a set of scenarios.

    Attributes:
        data (xr.Dataset): A dataset representing the scenario data.
        clustering (sklearn.cluster.KMedoids): A clustering model used to reduce the number of scenarios.
    """

    def __init__(self, data: dict[pd.DataFrame]) -> None:
        """
        Initializes a new instance of the ScenarioSet class.

        Args:
            data (dict[pd.DataFrame]): A dictionary of pandas DataFrames representing the scenario data.
        """
        self.data = xr.Dataset(
            {name: (("time", "scenario"), df.values) for name, df in data.items()},
            coords={
                "time": list(data.values())[0].index,
                "scenario": list(data.values())[0].columns,
            },
        )

    @property
    def normalised(self):
        """
        Returns a normalised version of the scenario data.

        Returns:
            xr.Dataset: The normalised dataset.
        """
        return self.data.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))

    def denormalise(self, ds_normalised: xr.Dataset) -> xr.Dataset:
        """
        Denormalises the given dataset using the min-max scaling method.

        Args:
            ds_normalised (xr.DataSet): The normalised dataset to denormalise.

        Returns:
            xr.Dataset: The denormalised dataset.
        """
        return ds_normalised.apply(
            lambda x: x * (self.data[x.name].max() - self.data[x.name].min()) + self.data[x.name].min()
        )

    @staticmethod
    def to_array(ds):
        """
        Converts a xarray Dataset to a stacked xarray DataArray.

        Parameters:
            ds (xarray.Dataset): The dataset to convert.

        Returns:
            xarray.DataArray: The stacked data array.
        """
        da = xr.concat([ds[var] for var in ds.data_vars], dim="var")
        da["var"] = list(ds.data_vars)
        return da.stack(row=("var", "time"))

    def reconvert(self, arr: np.ndarray, row) -> xr.Dataset:
        """
        Converts a numpy array to an xarray dataset.

        Parameters:
            arr (np.ndarray): The numpy array to be converted.
            row (list): A list of row labels.

        Returns:
            xr.Dataset: The converted xarray dataset.
        """
        da = (
            xr.DataArray(arr, dims=["col", "row"]).assign_coords(row=row).unstack("row")
        )

        return da.to_dataset(dim="var")

    @staticmethod
    def _cluster(value: np.ndarray, n_clusters) -> KMedoids:
        """
        Clusters the given data using K-Medoids algorithm.

        Args:
            value (numpy.ndarray): The data to be clustered.
            n_clusters (int): The number of clusters to form.

        Returns:
            sklearn.cluster.KMedoids: The clustering model.
        """
        clustering = KMedoids(metric="euclidean", n_clusters=n_clusters).fit(value)

        return clustering

    def reduce(self, n_clusters):
        """
        Reduce the number of scenarios by clustering the scenarios and returning the cluster centers.

        Args:
            n_clusters (int): number of clusters to reduce to
        Returns:
            xr.Dataset: reduced scenarios
        """
        # convert to 2-d array by stacking data
        arr = ScenarioSet.to_array(self.normalised)
        # cluster
        self.clustering = ScenarioSet._cluster(
            arr, n_clusters
        )
        # get cluster centers
        centers = self.clustering.cluster_centers_
        # reconvert to data set
        reconverted = self.reconvert(xr.DataArray(centers), row=arr.row)
        # denormalise and return
        return self.denormalise(reconverted)

    @property
    def scenario_probabilities(self):
        """
        Computes the probability of each scenario belonging to each cluster.

        Returns:
            probabilities : numpy.ndarray
                A 1D array of length n_clusters, where each element represents the
                probability of a scenario belonging to the corresponding cluster.
        """
        n_clusters = len(self.clustering.cluster_centers_)
        n_scenarios = len(self.clustering.labels_)

        return (
            np.array(
                [
                    np.sum([i == n for i in self.clustering.labels_.tolist()])
                    for n in range(n_clusters)
                ]
            )
            / n_scenarios
        )
    