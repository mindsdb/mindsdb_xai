import json
from typing import Dict, Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from mindsdb_xai.helpers import log


class DataInsight:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.insight = {}

    def explain(self, df: pd.DataFrame) -> Dict[str, object]:
        raise NotImplementedError

    def visualize(self, df: Optional[pd.DataFrame], show: bool = False) -> None:
        raise NotImplementedError

    def export(self) -> str:
        return json.dumps(self.insight)


class DataCorrelation(DataInsight):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "insight_template" in kwargs:
            self.insight = kwargs["insight_template"]
        else:
            self.insight = {
                "correlation_matrix": None,
                "column_names": None
            }

    def explain(self, df: pd.DataFrame) -> Dict[str, object]:
        log.info(f"{self.__class__.__name__} - Generating insight...")  # noqa
        self.insight["correlation_matrix"] = df.corr()
        self.insight["column_names"] = df.columns
        return self.insight

    def visualize(self, df: Optional[pd.DataFrame] = None, show: bool = False) -> None:
        log.info(f"{self.__class__.__name__} - Visualizing insight...")  # noqa
        if df is not None:
            self.explain(df)
        corrs = self.insight['correlation_matrix']

        cm = sns.heatmap(
            corrs,
            annot=True,
            center=0.0,
            vmin=-1.0,
            vmax=1.0,
            cmap=sns.diverging_palette(20, 220, as_cmap=True),
        )

        if show:
            plt.show()

        return cm
