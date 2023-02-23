from typing import Dict, Optional

import pandas as pd
import seaborn as sns
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
                "column_names": None
            }

    def explain(self, df: pd.DataFrame) -> Dict[str, object]:
        log.info(f"{self.__class__.__name__} - Generating insight...")  # noqa
        self.insight["correlogram"] = sns.pairplot(df)
        return self.insight

    def visualize(self, df: Optional[pd.DataFrame] = None, show: bool = False) -> None:
        log.info(f"{self.__class__.__name__} - Visualizing insight...")  # noqa
        if df is not None:
            self.explain(df)
        corr = self.insight['correlogram']

        if show:
            plt.show(corr)

        return corr
