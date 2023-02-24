import unittest

import pandas as pd

from mindsdb_xai.data.corr import DataCorrelation


class TestDataCorrelation(unittest.TestCase):
    def test_data_correlation(self):
        df = pd.read_csv("https://raw.githubusercontent.com/mindsdb/benchmarks/main/benchmarks/datasets/hdi/data.csv")
        explainer = DataCorrelation()
        explanation = explainer.explain(df)

        self.assertTrue(isinstance(explanation, dict))
        self.assertEqual(list(explanation.keys()), ['correlation_matrix', 'column_names'])
        self.assertTrue(isinstance(explanation['column_names'], list))
        self.assertTrue(isinstance(explanation['correlation_matrix'], pd.DataFrame))

        seralized = explainer.export()
        self.assertTrue(isinstance(seralized, str))

        explainer.visualize()
