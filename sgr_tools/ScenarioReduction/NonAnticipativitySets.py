
class NonAnticipativitySets:
    """
    Class to store non-anticipativity sets for a scenario tree.
    
    Args:
        sets_at_idx (dict): dict of non-anticipativity sets at each time index
        dict: {t: [set1, set2, ...]} where set1, set2, ... are sets of scenario indices
        empty (bool): if True, the NonAnticipativitySets object is empty and can be filled later
    Attributes:
        is_empty (bool): True if the NonAnticipativitySets object is empty
        """
    def __init__(self, sets_at_idx: dict = {}, empty=False):
        if not empty and len(sets_at_idx) < 1:
            raise ValueError(
                "sets_at_idx must be a non-empty dict or empty=True must be set"
            )

        self._sets_at_idx = sets_at_idx
        self._empty = empty

    def get_sets(self, t: int, s: int) -> set:
        """Return list of all scenarios indices which share a non-anticipativity set with s at time t

        Args:
            t (int): time index
            s (int): scenario index 
        """

        sets_at_idx = self._sets_at_idx[t]
        ret_val = []
        for val in sets_at_idx:
            if s in val:
                ret_val += val
        # return set to remove duplicates
        ret_val = set(ret_val)
        if len(ret_val) == 0:
            raise ValueError("No non-anticipativity sets found for scenario %s at time %s" % (s, t))

        return ret_val

    @property
    def is_empty(self) -> bool:
        return self._empty
