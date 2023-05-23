from scipy.stats import wilcoxon


def check_if_x_is_significantly_greater_than_y(x, y):
    """
    The p-value indicates if the distributions are significantly different or not by examining the tails. Specifying
    `greater` as alternative parameter, it compares if x > y. The null hypothesis considers the distributions equals.
    If the p-value is less than the confidence, you can say that x > y with that degree of confidence.
    """

    wilcox_v, p_value = wilcoxon(x, y, alternative='greater', zero_method='wilcox', correction=False)
    return wilcox_v, p_value
