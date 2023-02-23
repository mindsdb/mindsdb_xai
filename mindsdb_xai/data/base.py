import json
from typing import Dict, Optional

import pandas as pd


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
