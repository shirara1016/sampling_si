import numpy as np
from scipy.special import logsumexp
from collections.abc import Callable


class Pearson():
    def __init__(self, a):
        self.am = a - np.mean(a)
        self.aa = np.sum(self.am ** 2) ** 0.5

    def get(self, b):
        bm = b - np.mean(b)
        bb = np.sum(bm ** 2) ** 0.5
        ab = np.sum(self.am * bm)
        return np.abs(ab / (self.aa * bb))


class SMC():
    def __init__(
            self,
            pdf: Callable[[np.ndarray], np.ndarray],
            start: float = -100,
            end: float = 100,
            threshold: float = 0.5,
            correlation_threshold: float = 0.01,
            correlation_ratio: float = 0.9,
            seed: int | None = None):
        """Sequential Monte Carlo Sampler for one-dimensional random variables.

        Args:
            pdf (Callable[[np.ndarray], np.ndarray]):
                The pdf function of the one-dimensional target distribution.
                It should be a universal function so that
                it can take np.ndarray as an argument.
            start (float, optional):
                The starting point for sampling from a uniform distribution at
                infinite temperature. Defaults to -100.
            end (float, optional): _description_. Defaults to 100.
                The ending point for sampling from a uniform distribution at
                infinite temperature. Defaults to -100.
            threshold (float, optional):
                The threshold for the rate of decrease in ESS allowed during
                inverse temperature update. Defaults to 0.5.
            correlation_threshold (float, optional):
                The threshold of the rate of decrease at which
                the autocorrelation is  considered to have reached a
                stable state in Metropolis Hasting. Defaults to 0.01.
            correlation_ratio (float, optional):
                The threshold for the percentage of steady state in
                terminating Metropolis Hastings. Defaults to 0.9.
            seed (int | None, optional):
                Random generator initialization seed. Defaults to None.
        """

        def logpdf(x: np.ndarray) -> np.ndarray:
            return np.log(pdf(x))
            # return np.log(pdf(x) + 1e-15)

        self.logpdf = logpdf
        self.threshold = threshold
        self.correlation_threshold = correlation_threshold
        self.correlation_ratio = correlation_ratio
        self.rng = np.random.default_rng(seed=seed)

        self.reset(start, end)

    def reset(self, start: float | None = None, end: float | None = None):
        """Reset Sequential Monte Carlo Sampler.

        Args:
            start (float | None, optional):
                The starting point can be re-specified. Defaults to None.
            end (float | None, optional):
                The ending point can be re-specified. Defaults to None.
        """

        self.beta = 0.0
        self.weights = None
        self.start = start if start is not None else self.start
        self.end = end if end is not None else self.end

    def _update_beta_and_weights(self):
        old_beta = self.beta
        low_beta = self.beta
        high_beta = 2.0

        while high_beta - low_beta > 1e-6:
            new_beta = (low_beta + high_beta) / 2
            log_weights_un = (new_beta - old_beta) * self.loglikelihood
            log_weights = log_weights_un - logsumexp(log_weights_un)

            ESS = int(np.exp(-logsumexp(2 * log_weights)))
            if ESS == self.ESS_threshold:
                break
            elif ESS < self.ESS_threshold:
                high_beta = new_beta
            else:
                low_beta = new_beta

        if new_beta >= 1:
            new_beta = 1.0
            log_weights_un = (new_beta - old_beta) * self.loglikelihood
            log_weights = log_weights_un - logsumexp(log_weights_un)

        self.beta = new_beta

        self.weights = np.exp(log_weights)
        self.weights /= np.sum(self.weights)

    def _resample(self):
        indexes = self.rng.choice(
            self.num_samples, self.num_samples, p=self.weights)

        self.x = self.x[indexes]
        self.loglikelihood = self.loglikelihood[indexes]

    def _mutate(self):
        old_corr = 2.0
        corr = Pearson(self.x)
        while True:
            log_R = np.log(self.rng.random(self.num_samples))
            proposal = self.rng.normal(self.x, 1, self.num_samples)
            proposal_lp = self.logpdf(proposal)

            accepted = log_R < self.beta * \
                (proposal_lp - self.loglikelihood)

            self.x[accepted] = proposal[accepted]
            self.loglikelihood[accepted] = proposal_lp[accepted]

            pearson_r = corr.get(self.x)
            ratio = np.mean(
                (old_corr - pearson_r) > self.correlation_threshold)
            if ratio > self.correlation_ratio:
                old_corr = pearson_r
            else:
                break

    def sampling(self, num_samples: int) -> np.ndarray:
        """Sampling from target distribution.

        Args:
            num_samples (int):
                Number of samples to be sampled from the target distribution.

        Returns:
            np.ndarray:
                Random numbers sampled from the target distribution.
        """
        self.num_samples = num_samples
        self.ESS_threshold = int(self.num_samples * self.threshold)
        self.x = self.rng.uniform(self.start, self.end, self.num_samples)
        self.loglikelihood = self.logpdf(self.x)

        while True:
            self._update_beta_and_weights()
            self._resample()
            self._mutate()
            if self.beta >= 1:
                break

        return self.x.copy()
