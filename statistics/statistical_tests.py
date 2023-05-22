from scipy.stats import wilcoxon


def check_if_x_is_significantly_greater_than_y(x, y):

    wilcox_v, p_value = wilcoxon(x, y, alternative='greater', zero_method='wilcox', correction=False)
    return wilcox_v, p_value
