import numpy as np
from scipy.stats import norm, chi2


def indicator(intervals):
    intervals = intervals

    def flag(x):
        for lower, upper in intervals:
            if lower <= x <= upper:
                return 1.0
        return 0.0
    return np.vectorize(flag)


def tn_pdf(x, intervals):
    mask = indicator(intervals)
    normalize = 0
    for lower, upper in intervals:
        normalize += (norm.cdf(upper) - norm.cdf(lower))
    return norm.pdf(x) * mask(x) / normalize


def tn_cdf(x, intervals):
    denom = 0
    nom = 0
    for lower, upper in intervals:
        item = norm.cdf(upper) - norm.cdf(lower)
        denom += item
        if upper <= x:
            nom += item
        elif lower <= x <= upper:
            nom += (norm.cdf(x) - norm.cdf(lower))
    return nom / denom


def tc2_pdf(x, v, intervals):
    mask = indicator(intervals)
    normalize = 0
    for lower, upper in intervals:
        normalize += (chi2.cdf(upper, v) - chi2.cdf(lower, v))
    return chi2.pdf(x, v) * mask(x) / normalize


def tc2_cdf(x, v, intervals):
    denom = 0
    nom = 0
    for lower, upper in intervals:
        item = chi2.cdf(upper, v) - chi2.cdf(lower, v)
        denom += item
        if upper <= x:
            nom += item
        elif lower <= x <= upper:
            nom += (chi2.cdf(x, v) - chi2.cdf(lower, v))
    return nom / denom
