import json
from typing import Dict, Optional

import pandas as pd


class DataInsight:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.insight = {}
        self.visualization_params = {}  # only serializable values

    def explain(self, df: pd.DataFrame) -> Dict[str, object]:
        raise NotImplementedError

    def visualize(self, df: Optional[pd.DataFrame]) -> None:
        raise NotImplementedError

    def export(self) -> str:
        return json.dumps({
            "data": self.insight,
            "params": self.visualization_params
        })
