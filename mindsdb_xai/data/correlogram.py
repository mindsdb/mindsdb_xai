from typing import Dict, Optional
from itertools import combinations

import pandas as pd
import matplotlib.pyplot as plt

from mindsdb_xai.helpers import log
from mindsdb_xai.data.base import DataInsight


class DataCorrelogram(DataInsight):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "insight_template" in kwargs:
            self.insight = kwargs["insight_template"]
        else:
            self.insight = {
                "correlogram": None,
                "histograms_columns_names": None
            }

    def explain(self, df: pd.DataFrame) -> Dict[str, object]:
        log.info(f"{self.__class__.__name__} - Generating insight...")  # noqa

        # diagonal histograms
        hists = []
        self.insight['cols'] = list(df.columns)
        for col in df.columns:
            hists.append(df[col].hist())
        self.insight['hists'] = hists

        # scatters
        combs = list(combinations(df.columns, 2))
        scatters = []
        for comb in combs:
            scatters.append(plt.scatter(df[comb[0]], df[comb[1]]))
        self.insight['scatters'] = scatters

        return self.insight

    def visualize(self, df: Optional[pd.DataFrame] = None) -> None:
        log.info(f"{self.__class__.__name__} - Visualizing insight...")  # noqa
        if df is not None:
            self.explain(df)

        cols = self.insight['cols']
        hists = self.insight['hists']
        scatters = self.insight['scatters']

        raise NotImplementedError
