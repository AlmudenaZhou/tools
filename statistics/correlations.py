import numpy as np
from scipy import stats


def correlation_pointbiserialr(x, y):
    return stats.pointbiserialr(x, y).correlation


def correlation_info(df, corr_method: str = 'pearson', corr_threshold: float = 0.8, missing_suffix: str = '_NA'):
    all_corr = df.corr(method=corr_method)
    corr_missing_data = all_corr[all_corr.index.str.endswith(missing_suffix)]
    corr_missing_data.replace(1.0, np.nan, inplace=True)
    return (corr_missing_data[corr_missing_data.abs() > corr_threshold]
            .dropna(axis=1, how='all')
            .dropna(axis=0, how='all'))
