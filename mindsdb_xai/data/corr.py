import json
import importlib
from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go

from mindsdb_xai.helpers import log
from mindsdb_xai.data.base import DataInsight


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

        self.visualization_params = {
            'data': {
                'texttemplate': "%{z}",
                'textfont': {"size": 20},
                'colorscale': 'blackbody'
            },
            'layout': {
                'title': {
                    "text": "Correlation Matrix",
                    "x": 0.5,
                },
                'xaxis': {
                    "title": "Columns"
                },
                'yaxis': {
                    "title": "Columns",
                    "autorange": "reversed",
                },
            },
            'module': 'plotly.graph_objects',
            'class': 'Heatmap'
        }

    def explain(self, df: pd.DataFrame) -> Dict[str, object]:
        log.info(f"{self.__class__.__name__} - Generating insight...")  # noqa
        self.insight["correlation_matrix"] = df.corr()
        self.insight["column_names"] = list(df.columns)
        return self.insight

    def visualize(self, df: Optional[pd.DataFrame] = None) -> None:
        log.info(f"{self.__class__.__name__} - Visualizing insight...")  # noqa
        if df is not None:
            self.explain(df)

        viz_class = getattr(importlib.import_module(self.visualization_params['module']),
                            self.visualization_params['class'])
        data = viz_class(
            text=self.insight['correlation_matrix'],
            z=self.insight['correlation_matrix'],
            x=self.insight['column_names'],
            y=self.insight['column_names'],
            **self.visualization_params['data']
        ),

        layout = go.Layout(**self.visualization_params['layout'])
        fig = go.Figure(data=data, layout=layout)
        fig.show()

    def export(self) -> str:
        return json.dumps({
            "data": {
                "correlation_matrix": self.insight['correlation_matrix'].to_json(),
                "column_names": self.insight['column_names'],
            },
            "params": self.visualization_params
        })
