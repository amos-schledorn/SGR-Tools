import pandas as pd
import numpy as np
import patsy


import pandas as pd
import numpy as np
import patsy

class Splines:
    """
    A class used to generate periodic bsplines

    Attributes
    ----------
    seasonality : int
        The seasonality of the data
    degrees_freedom : int, optional
        The degrees of freedom for the splines, by default 4
    model_weekend : bool, optional
        Whether to model weekend splines, by default True

    Methods
    -------
    get_fit(idx: pd.DatetimeIndex) -> pd.DataFrame
        Generate periodic bsplines
    """
    def __init__(
        self, seasonality: int, degrees_freedom: int = 4, model_weekend: bool = True
    ):
        """
        Parameters
        ----------
        seasonality : int
            The seasonality of the data
        degrees_freedom : int, optional
            The degrees of freedom for the splines, by default 4
        model_weekend : bool, optional
            Whether to model weekend splines, by default True
        """
        self.seasonality = seasonality
        self.degrees_freedom = degrees_freedom
        self.model_weekend = model_weekend

    def get_fit(self, idx: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Generate periodic bsplines

        Knots correspond to seasonality.

        Parameters
        ----------
        idx : pd.DatetimeIndex
            The index of the data

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the splines
        """
        # knots
        x = [np.mod(i, self.seasonality) for i in range(len(idx))]
        # splines
        splines = patsy.dmatrix(
            "cc(x, df=dgr) - 1", {"x": x, "dgr": self.degrees_freedom}
        )

        if self.model_weekend:
            # weekday splines
            splines_weekday = pd.DataFrame(
                splines * np.array(idx.weekday < 5).reshape(-1, 1),
                index=idx,
                columns=[f"bs-weekday_{i}" for i in range(self.degrees_freedom)],
            )
            # weekend splines
            splines_weekend = pd.DataFrame(
                splines * np.array(idx.weekday >= 5).reshape(-1, 1),
                index=idx,
                columns=[f"bs-weekend_{i}" for i in range(self.degrees_freedom)],
            )
            return pd.concat([splines_weekday, splines_weekend], axis=1)
        else:
            return pd.DataFrame(splines, index=idx)
