import numpy as np
from typing import Any, List
from sklearn_extra.cluster import KMedoids
import logging
import warnings
from . import NonAnticipativitySets

logger = logging.getLogger(__name__)


class ScenarioTree:
    """Build scenario tree for time series data recursively by reducing scenarios top-down with K-Medoids clustering and back-propagating results bottom-up.

    Args:
        data (np.ndarray): Time series data of shape (S, T) where S is the number of scenarios and T is the number of time steps. 
        split_idx (np.ndarray): Array of time steps at which to split the data. Must be strictly increasing and contain 0.
        split_branches (np.ndarray): Array of number of branches to split into at each time step specified in split_idx. Must have same length as split_idx.
        base_idx: Index of first time step in data. Used to keep track of time steps when splitting recursively. Defaults to 0. 
    
    Attributes:
        S (int): Number of scenarios.
        T (int): Number of time steps.
        split_idx (np.ndarray): Array of time steps at which to split the data. Must be strictly increasing and contain 0.
        split_branches (np.ndarray): Array of number of branches to split into at each time step specified in split_idx. Must have same length as split_idx.
        base_idx: Index of first time step in data. Used to keep track of time steps when splitting recursively. Defaults to 0.
        is_reduced (bool): Whether the scenario tree is reduced to a single scenario.
        width (int): Number of scenarios in the widest branch of the scenario tree.
        depth (int): Number of time steps in the scenario tree.
        is_leaf (bool): Whether the scenario tree is a leaf (i.e. has no children).
        children (List[ScenarioTree]): List of children of the scenario tree.
        parent (ScenarioTree): Parent of the scenario tree.
    """

    def __init__(
        self,
        data: np.ndarray,
        split_idx: np.ndarray,  # array-like
        split_branches: np.ndarray,  # array-like
        base_idx: int=0,
        seed: int= 1494):

        # np.random.seed(seed)
        self._seed = seed
        self._data = data
        self._T = self._data.shape[1]
        self._children = []

        self._validate_input(split_idx=split_idx, split_branches=split_branches)

        self.split_idx = np.array(split_idx)
        self.split_branches = np.array(split_branches)
        self.base_idx = base_idx


        logging.debug(
            "Creating scenario tree from %s scenarios and %s time steps with base index %s...",
            self.S,
            self.T,
            self.base_idx
        )
        logging.debug(
            "Input is to split at time steps %s into %s scenarios",
            self.split_idx,
            self.split_branches,
        )

        self.is_reduced = False
        self._prob = 1.0
        self._generate()

    @staticmethod
    def _validate_input(split_idx: np.ndarray, split_branches: np.ndarray) -> None:
        """
        Validates the input parameters for the ScenarioTree constructor.

        Parameters
        ----------
        split_idx : numpy.ndarray
            An array of time steps at which to split the scenario tree.
        split_branches : numpy.ndarray
            An array of the number of branches to create at each split.

        Raises
        ------
        ValueError
            If 0 is not in split_idx or split_idx is not strictly increasing.
            If split_idx and split_branches do not have the same length.
        """
        if len(split_idx) > 0 or len(split_branches) > 0:
            if len(split_idx) != len(split_branches):
                raise ValueError("split_idx and split_branches must have same length")
            if 0 not in split_idx:
                raise ValueError("0 must be in split_idx")
            if not np.all(np.diff(split_idx) > 0):
                raise ValueError("split_idx must be strictly increasing")
            if len(split_idx) != len(split_branches):
                raise ValueError("split_idx and split_branches must have same length")

    def _generate(self):
        """Generate senario tree recursively."""

        assert len(self.split_idx) == len(
            self.split_branches
        ), "split_idx and split_branches must have same length"

        # if still splits to be made
        # (when creating child, we pass self.split_idx[1:] to child)
        if self.split_idx.size > 0:
            # branch and add children (which are trees that are generated the same way)
            self._branch(self.split_idx[0], self.split_branches[0])
        # if intended to be leaf, reduce to one scenario (make leaf) and stop
        else:
            self._reduce_to_one_scenario()

    def _branch(self, t: float, n_branches: int) -> None:
        """Branch out at time step <t> into <n_branches> branches.

        Args:
            t (float): Time step at which to branch out.
            n_branches (int): Number of branches to split into.
        """

        logging.debug(
            "Splitting %s  at time step %s into %s branches...",
            self,
            t,
            n_branches,
        )

        # input value tests
        assert t >= 0, "t must be positive"
        assert t <= self.T, "t must be <= than data length"
        assert n_branches > 0, "n_branches must be positive"
        if n_branches > self.S:
            logging.warning("n_branches=%s > data size=%s, setting n_branches=%s", n_branches, self.S, self.S)
            n_branches = self.S

        clustering = self._cluster_scenarios(data=self._data, n_scenarios=n_branches)
        # add children
        self._add_children(
            [
                ScenarioTree(
                    # split_idx is relative, so pass data from next split onwards
                    # reduce idx of that split to zero and all remaining split_idx by same number
                    # example: t = 0 and next split should be made at t=5, so pass data from t=5 onwards, indicating next split to be made at t=0 (i.e. first time idx of that reduced data)
                    split_idx=self.split_idx[1:] - self._next_split,
                    split_branches=self.split_branches[1:],
                    data=self._data.copy()[
                        [
                            s
                            for s in range(self.S)
                            if clustering.labels_[s]
                            == n  # add data to child that is clustered to medoid
                        ],
                        self._next_split :,  # add data from t onwards
                    ],
                    base_idx=self.base_idx + self._next_split # add base_idx to child
                )
                for n, prob in enumerate(clustering.probs)  # loop over clusters
            ]
        )

        # update to reduced data
        self._data = clustering.cluster_centers_[:, : self._next_split]
        logging.debug("Updated tree data to array of shape %s", self._data.shape)
        # self._prob is either 1.0 (if not reduced or leaf) or scenario probabilities
        # get_prob adjusts for this by propagating probabilities upward
        self._prob = clustering.probs
        self.is_reduced = True

    def _add_children(self, children: List) -> None:
        """Add list of children to scenario tree.
        Args:
            children (List[ScenarioTree]): List of children to add."""
        logging.debug("Adding %s children to scenario tree...", len(children))
        self._children.extend(children)

    def _get_non_anticipativity_sets_at_nodes(self) -> dict:
        """Return non-anticipativity sets for each time step."""

        ret_val = {}  # non-anticipativity sets at splits
        for t, children in self.tree.items():
            ret_val[t] = []
            idx = 0
            for child in children:
                assert t == child.base_idx, "base_idx of child must be equal to time step"
                ret_val[t].append([i + idx for i in range(child.width)])
                idx += child.width
        
        return ret_val
        

    def _get_children_at_depth(self, depth: int) -> List:    
        """Return list of children at given depth.

        Args:
            depth (int): Depth at which to return children. 
        """
        if depth == 0:
            return [self]
        else:
            ret_val = []
            for child in self._children:
                ret_val.extend(child._get_children_at_depth(depth - 1))
            return ret_val
    
    @property
    def tree(self) -> dict:
        """Return tree as dictionary indexed by time."""
        return {t: self._get_children_at_depth(d) for d, t in enumerate(self.split_idx)}

    @property
    def depth(self) -> int:
        """Return depth of tree."""
        return len(self.split_idx)

    def _get_non_anticipativity_sets_at_idx(self) -> dict:
        """Return non-anticipativity sets for each time step."""
        sets_at_nodes = self._get_non_anticipativity_sets_at_nodes()
        sets_at_idx = {}  # non-anticipativity sets in time step
        sets_at_idx[self._T] = [[i] for i in range(self.width)]
        t = self._T - 1
        while t >= 0:
            # if split at t, add new non-anticipativity set
            if t in sets_at_nodes.keys():
                sets_at_idx[t] = sets_at_nodes[t]
            # else add set from preceding time step
            else:
                sets_at_idx[t] = sets_at_idx[t + 1]
            t -= 1
        
        return sets_at_idx

    def get_non_anticipativity_sets(self):
        """Return non-anticipativity sets for each time step as NonAnticipativitySets object."""
        return NonAnticipativitySets(self._get_non_anticipativity_sets_at_idx())

    @property
    def _next_split(self) -> float:
        """Return next split time step."""
        return np.max(self.split_idx[:2])

    def _reduce_to_one_scenario(self) -> None:
        """Reduce scenario tree to one scenario."""
        logging.debug("Reducing %s into to one scenario...", self)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="n_clusters should be larger than 2 if max_iter != 0 setting max_iter to 0.")
            clustering = self._cluster_scenarios(data=self._data, n_scenarios=1)
        self._data = clustering.cluster_centers_
        logging.debug("Update tree data to array of shape %s", self._data.shape)
        self.is_reduced = True

    def _cluster_scenarios(self, data: np.ndarray, n_scenarios: int) -> np.ndarray:
        """Reduce scenarios to medoids of K-Medoids clustering.

        Args:
            data (np.ndarray): Data to cluster.
            n_scenarios (int): Number of scenarios to cluster to. 
        """

        # K-Medoids
        clustering = KMedoids(metric="euclidean", n_clusters=n_scenarios, random_state=self._seed, method='pam').fit(data)
        # Scenario probability is ratio of scenario points in cluster
        clustering.probs = (
            np.array(
                [
                    np.sum([i == n for i in clustering.labels_.tolist()])
                    for n in range(n_scenarios)
                ]
            )
            / data.shape[0]
        )

        return clustering

    def get_scenario_data(self) -> np.ndarray:
        """Get data of scenario tree. If reduced, return reduced scenarios."""

        if self.is_reduced:
            if self.is_leaf:
                logging.debug(
                    "Returning reduced data of shape %s:", self._data.flatten().shape
                )
                return self._data.flatten()
            else:
                # initialise return array
                ret_val = np.empty((self.width, self._T))
                # loop over children
                logging.debug("Returning data of %s: %s", self, self._data)

                start_row = 0
                for s, child in enumerate(self._children):
                    logging.debug(
                        "Adding child %s with width %s to return array",
                        child,
                        child.width,
                    )
                    end_row = start_row + child.width

                    ret_val[
                        start_row:end_row, : self._next_split
                    ] = self._data[s]
                    ret_val[
                        start_row : end_row,
                        self._next_split :,
                    ] = child.get_scenario_data()
                   
                    start_row = end_row

                return ret_val

        else:
            logging.debug("Returning data of %s:", self._data)
            return self._data

    def get_scenario_probabilities(self):
        "Compute scenario probabilities"
        # if leaf, return 1.0
        if self.is_leaf:
            return self._prob
        else:
            ret_val = np.empty(self.width)  # self.width = no. of scenarios
            # get child probabilities recursively
            start_idx = 0
            for s, child in enumerate(self._children):
                end_idx = start_idx + child.width
                ret_val[start_idx:end_idx] = self._prob[s]
                ret_val[
                    start_idx:end_idx
                ] *= child.get_scenario_probabilities()
                start_idx = end_idx
            return ret_val

    @property
    def T(self) -> int:
        """Number of time steps in data."""
        return self._data.shape[1]

    @property
    def S(self) -> int:
        """Number of scenarios in data."""
        return self._data.shape[0]

    @property
    def depth(self) -> int:
        """Depth of scenario tree."""
        if self.is_leaf:
            return 0
        else:
            return 1 + max([child.depth for child in self._children])

    @property
    def width(self) -> int:
        """Width of scenario tree."""
        if self.is_leaf:
            return self.S
        else:
            return np.sum([child.width for child in self.children])

    @property
    def is_leaf(self) -> bool:
        """Check if scenario tree is leaf."""
        return not self._children

    @property
    def children(self):
        """Children of scenario tree."""
        return self._children

    @property
    def parent(self):
        """Parent of scenario tree."""
        return self._parent

    def __str__(self) -> str:
        if self.is_reduced:
            if self.is_leaf:
                return f"Leaf of {self.S} scenarios and {self.T} time steps"
            else:
                return f"Scenario tree reduced to {self.S} scenarios and {self.T} time steps"
        else:
            return f"Raw scenario tree of {self.S} scenarios and {self.T} time steps"

    def __repr__(self) -> str:
        return self.__str__()
