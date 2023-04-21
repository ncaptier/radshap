import warnings
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


class Shapley:
    """
    Parameters
    ----------

    Returns
    -------
    """

    def __init__(
        self,
        ninstances,
        nsamples,
        predictor,
        batch_creator,
        empty_value=0.5,
        estimator="auto",
        n_jobs=1,
    ):
        self.ninstances = ninstances
        self.nsamples = nsamples
        self.predictor = predictor
        self.batch_creator = batch_creator
        self.empty_value = empty_value
        self.estimator = estimator
        self.n_jobs = n_jobs

        self.shapleyvalues_ = None

    def fit(self, X, **kwargs):

        if self.estimator == "auto":
            if self.ninstances <= 7:
                return self._fit_exact(X, **kwargs)
            else:
                return self._fit_antithetic(X, **kwargs)
        elif self.estimator == "exact":
            if self.ninstances > 7:
                warnings.warn(
                    "The number of permutations to test for exact estimation is greater than 40320."
                    " It may require some computational power."
                )
            return self._fit_exact(X, **kwargs)
        elif self.estimator == "antithetic":
            return self._fit_antithetic(X, **kwargs)
        else:
            raise ValueError("estimator ")

    def _fit_antithetic(self, X, **kwargs):
        results = np.zeros((self.nsamples // 2, self.ninstances))
        parallel = Parallel(n_jobs=self.n_jobs, verbose=0)
        sampling_results = parallel(
            delayed(self._get_antithetic_evaluations)(X, **kwargs)
            for _ in range(self.nsamples // 2)
        )

        for count, value in enumerate(sampling_results):
            results[count, value[0]] = value[1]

        self.shapleyvalues_ = (1 / self.nsamples) * np.sum(results, axis=0)
        return self

    def _get_antithetic_evaluations(self, X, **kwargs):
        p = np.random.permutation(np.arange(self.ninstances))
        marginals = _get_evaluations(
            p, self.batch_creator, self.predictor, self.empty_value, X, **kwargs
        )[1]

        p_r = np.flip(p)
        marginals_r = np.flip(
            _get_evaluations(
                p_r, self.batch_creator, self.predictor, self.empty_value, X, **kwargs
            )[1]
        )
        return p, marginals + marginals_r

    def _fit_exact(self, X, **kwargs):
        n = np.math.factorial(self.ninstances)
        results = np.zeros((n, self.ninstances))
        parallel = Parallel(n_jobs=self.n_jobs, verbose=0)
        exact_results = parallel(
            delayed(_get_evaluations)(
                perm=p,
                batch_creator=self.batch_creator,
                predictor=self.predictor,
                empty_value=self.empty_value,
                X=X,
                **kwargs
            )
            for p in _get_permutations(self.ninstances)
        )

        for count, value in enumerate(exact_results):
            results[count, value[0]] = value[1]

        self.shapleyvalues_ = (1 / n) * np.sum(results, axis=0)
        return self

    def plot(self, nbest=10, names=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        if names is not None:
            df = pd.DataFrame(self.shapleyvalues_, index=names, columns=["shapley"])
        else:
            df = pd.DataFrame(
                self.shapleyvalues_,
                index=["instance_" + str(i) for i in range(len(self.shapleyvalues_))],
                columns=["shapley"],
            )

        df["shapley_abs"] = np.abs(df["shapley"])
        df = df.sort_values(by="shapley_abs", ascending=False).iloc[
            : max(nbest, len(self.shapleyvalues_)), 0
        ]

        df.plot.barh(color=(df["shapley"] > 0).map({True: "red", False: "blue"}), ax=ax)
        return


def _get_permutations(n):
    for p in permutations(range(n)):
        yield np.array(p)


def _get_evaluations(perm, batch_creator, predictor, empty_value, X, **kwargs):
    batch = batch_creator(perm, X, **kwargs)
    values = predictor(batch)
    marginals = _get_marginals(values, empty_value)
    return perm, marginals


def _get_marginals(v, first_value):
    v_ = np.zeros(v.shape)
    v_[0] = first_value
    v_[1:] = v.copy()[0:-1]
    return v - v_
