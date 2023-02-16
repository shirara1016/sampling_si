import numpy as np
from scipy.stats import norm, chi2


def indicator(intervals):
    def flag(x):
        for lower, upper in intervals:
            if lower <= x <= upper:
                return True
        return False
    return np.vectorize(flag)


def tn_log_pdf(mask):
    def logpdf(x):
        cond = mask(x)
        lp = np.full(len(x), -np.inf)
        lp[cond] = - x[cond] ** 2 / 2
        return lp
    return logpdf


def tc2_log_pdf(mask, v):
    def logpdf(x):
        cond = mask(x) * (x > 0)
        lp = np.full(len(x), -np.inf)
        lp[cond] = (0.5 * v - 1) * np.log(x[cond]) - x[cond] / 2
        return lp
    return logpdf


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
