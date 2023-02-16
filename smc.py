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
            logpdf: Callable[[np.ndarray], np.ndarray],
            start: float = -100,
            end: float = 100,
            threshold: float = 0.5,
            correlation_threshold: float = 0.01,
            correlation_ratio: float = 0.9,
            seed: int | None = None):
        """Sequential Monte Carlo Sampler for one-dimensional random variables.

        Args:
            logpdf (Callable[[np.ndarray], np.ndarray]):
                The logarithm pdf function of the one-dimensional target distribution.
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

        self.logpdf = logpdf
        self.start = start
        self.end = end
        self.threshold = threshold
        self.correlation_threshold = correlation_threshold
        self.correlation_ratio = correlation_ratio
        self.rng = np.random.default_rng(seed=seed)

    def _reset(self):
        self.beta = 0.0
        self.weights = None
        self.iteration = 0

        self.mh_steps = []
        self.betas = []

    def _update_beta_and_weights(self):
        self.iteration += 1

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
        self.betas.append(self.beta)

        self.weights = np.exp(log_weights)
        self.weights /= np.sum(self.weights)

    def _tune(self):
        if self.iteration > 1:
            chain_scales = np.exp(
                np.log(self.proposal_scales) + (self.chain_acc_rate - 0.234))
            self.proposal_scales = 0.5 * (chain_scales + np.mean(chain_scales))
        ave = np.sum(self.x * self.weights)
        var = np.sum(self.weights * ((self.x - ave) ** 2))
        self.std = np.sqrt(var)

    def _resample(self):
        indexes = self.rng.choice(
            self.num_samples, self.num_samples, p=self.weights)

        self.x = self.x[indexes]
        self.loglikelihood = self.loglikelihood[indexes]

        if self.iteration > 1:
            self.proposal_scales = self.proposal_scales[indexes]
            self.chain_acc_rate = self.chain_acc_rate[indexes]

    def _mutate(self):
        old_corr = 2.0
        corr = Pearson(self.x)

        ac = []
        mh_step = 0

        while True:
            log_R = np.log(self.rng.random(self.num_samples))
            proposal = self.x + self.proposal_scales * self.std * \
                self.rng.normal(size=self.num_samples)
            proposal_lp = self.logpdf(proposal)

            accepted = log_R < self.beta * \
                (proposal_lp - self.loglikelihood)
            ac.append(accepted)

            self.x[accepted] = proposal[accepted]
            self.loglikelihood[accepted] = proposal_lp[accepted]

            mh_step += 1

            pearson_r = corr.get(self.x)
            ratio = np.mean(
                (old_corr - pearson_r) > self.correlation_threshold)
            if ratio > self.correlation_ratio:
                old_corr = pearson_r
            else:
                break

        self.chain_acc_rate = np.mean(ac, axis=0)
        self.mh_steps.append(mh_step)

    def sampling(self, num_samples: int) -> np.ndarray:
        """Sampling from target distribution.

        Args:
            num_samples (int):
                Number of samples to be sampled from the target distribution.

        Returns:
            np.ndarray:
                Random numbers sampled from the target distribution.
        """
        self._reset()

        self.num_samples = num_samples
        self.ESS_threshold = int(self.num_samples * self.threshold)
        self.x = self.rng.uniform(self.start, self.end, self.num_samples)
        self.loglikelihood = self.logpdf(self.x)

        self.proposal_scales = np.ones(self.num_samples)

        while True:
            self._update_beta_and_weights()
            self._tune()
            self._resample()
            self._mutate()
            if self.beta >= 1:
                break

        return self.x.copy()
