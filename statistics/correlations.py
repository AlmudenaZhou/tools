import numpy as np
from scipy import stats


def correlation_pointbiserialr(x, y):
    return stats.pointbiserialr(x, y).correlation


def correlation_info(df, corr_method='pearson', corr_threshold=0.8):
    all_corr = df.corr(method=corr_method)
    corr_missing_data = all_corr[all_corr.index.str.endswith('_is_missing')]
    corr_missing_data.replace(1.0, np.nan, inplace=True)
    return (corr_missing_data[corr_missing_data.abs() > corr_threshold]
            .dropna(axis=1, how='all')
            .dropna(axis=0, how='all'))
