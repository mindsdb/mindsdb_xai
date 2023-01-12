import os
import logging
import colorlog
import pandas as pd

from type_infer.helpers import get_ts_groups

def initialize_log():
    pid = os.getpid()
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter())

    logging.basicConfig(handlers=[handler])
    log = logging.getLogger(f'posthoc_xai-{pid}')
    log_level = os.environ.get('POSTHOC_XAI_LOG', 'DEBUG')
    log.setLevel(log_level)
    return log


log = initialize_log()


def filter_ds(df: pd.DataFrame, tss: dict, n_rows: int = 1):
    """
    This method only triggers for timeseries datasets.

    It returns a dataframe that filters out all but the first ``n_rows`` per group.
    """  # noqa
    if tss.get('is_timeseries', False):
        gby = tss.get('group_by', None)
        if gby is None:
            df = df.iloc[[0]]
        else:
            ndf = pd.DataFrame(columns=df.columns)
            for group in get_ts_groups(df, tss):
                if group != '__default':
                    _, subdf = get_group_matches(df, group, tss.group_by)
                    ndf = pd.concat([ndf, subdf.iloc[:n_rows]])
            df = ndf
    return df
