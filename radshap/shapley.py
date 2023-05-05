import warnings
from itertools import permutations
from typing import NoReturn, Optional, Tuple, Callable, Union, Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ._aggregator import get_batch_creator


class Shapley:
    """Computes the Shapley value of every element of a collection of instances that are aggregated together in a single
     input of a trained predictive algoirhtm.
    It either uses an exact enumeration strategy when the number of samples is relatively small (<8) or an approximate
    Monte Carlo scheme with antithetic sampling.

    Parameters
    ----------
    predictor: callable (input: 2D array of shape (n_inputs, n_input_features), output: 1D array of shape (n_inputs,))
        Trained predictor that returns one single real-value prediction per input.

    aggregation: callable, tuple, list of tuples
        Aggregator that transforms an array of n_instances (with each one characterized by n_instance_features) into one
        single vector of shape (n_input_featurs) that will be used as an input of the predictor.

        To define the aggregator one can use:

            * a callable that takes as input a 2D array of shape (n_instances, n_instance_features) and returns a 1D
            array of shape (1, n_input_features).
            * a tuple (method, subset) with method being a string that refers to a numpy aggregating function (e.g 'sum'
            , 'min', 'std'...) and subset being a 1D array that defines the subset of columns/features on which to apply
            this method (or None for applying it on all the columns/features).
            * a list of tuples [(method_1, subset_1), (method_2, subset_2), ...] to define several aggregators. Please
            note that the aggregated features will be ordered according to the order of the provided list (i.e
            [agg_feature_method_1, ..., agg_feature_method_2, ...]).

    empty_value: float, otpional
        The default is 0.5.

    Attributes
    -------
    shapleyvalues_: 1D array of shapes (n_instances,)
        Shapley value associated to each instance.

    Examples
    -------
    >>> import numpy as np
    >>> import joblib
    >>> from radshap.shapley import Shapley
    >>>
    >>> model = joblib.load("trained_logistic_regression.joblib")
    >>> shap = Shapley(predictor = lambda x: model.predict_proba(x)[:, 1], aggregation = ('mean', None))
    >>> shap.explain(X) # X a 2D array of shape (n_instances, n_instance_features)
    """

    def __init__(
        self,
        predictor: Callable[[np.ndarray], float],
        aggregation: Callable[[np.ndarray], np.ndarray],
        empty_value: Optional[float] = 0.5,
    ) -> NoReturn:

        self.predictor = predictor
        self.empty_value = empty_value

        self.batch_creator = get_batch_creator(aggregation)
        self._ninstances = None
        self.shapleyvalues_ = None

    def explain(
        self,
        X: np.ndarray,
        estimation_method: Optional[str] = "auto",
        nsamples: Optional[int] = 1000,
        n_jobs: Optional[int] = 1,
    ) -> np.ndarray:
        """Computes the Shapley values for each row of X.

        X must correspond to a valid collection of instances (shape (n_instances, n_instance_features)) that will be
        passed to the aggregator function and then used as an input to the predictor.

        Parameters
        ----------
        X: 2D array of shape (n_instances, n_instance_features)
            X corresponds to a collection of instances that will be aggregated into a single input

        estimation_method: str {'auto', 'exact', 'antithetic'}, optional
            Estimation method for the Shapley values. The default is "auto".

        nsamples: str, optional
            The default is 1000.

        n_jobs: int, optional
            Number of jobs to run in parallel. -1 means using all processors.
            See the joblib package documentation for more explanations. The default is 1.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._ninstances = X.shape[0]
        if estimation_method == "auto":
            if self._ninstances <= 7:
                return self._explain_exact(X, n_jobs)
            else:
                return self._explain_antithetic(X, nsamples, n_jobs)
        elif estimation_method == "exact":
            if self._ninstances > 7:
                warnings.warn(
                    "The number of permutations to test for exact estimation is at least 40320."
                    " It may require some computational power."
                )
            return self._explain_exact(X, n_jobs)
        elif estimation_method == "antithetic":
            return self._explain_antithetic(X, nsamples, n_jobs)
        else:
            raise ValueError(
                "Unrecognized estimation method. Please choose among 'auto', 'exact', or antithetic'"
            )

    def _explain_antithetic(self, X: np.ndarray, nsamples: int, n_jobs: int) -> np.ndarray:
        """ """
        results = np.zeros((nsamples // 2, self._ninstances))
        parallel = Parallel(n_jobs=n_jobs, verbose=0)
        sampling_results = parallel(
            delayed(self._get_antithetic_evaluations)(X) for _ in range(nsamples // 2)
        )

        for count, value in enumerate(sampling_results):
            results[count, value[0]] = value[1]

        self.shapleyvalues_ = (1 / nsamples) * np.sum(results, axis=0)
        return self.shapleyvalues_

    def _get_antithetic_evaluations(self, X: np.ndarray) -> object:
        """ """
        p = np.random.permutation(np.arange(self._ninstances))
        marginals = _get_evaluations(
            p, self.batch_creator, self.predictor, self.empty_value, X
        )[1]

        p_r = np.flip(p)
        marginals_r = np.flip(
            _get_evaluations(
                p_r, self.batch_creator, self.predictor, self.empty_value, X
            )[1]
        )
        return p, marginals + marginals_r

    def _explain_exact(self, X: np.ndarray, n_jobs: int) -> np.ndarray:
        """ """
        n = np.math.factorial(self._ninstances)
        results = np.zeros((n, self._ninstances))
        parallel = Parallel(n_jobs=n_jobs, verbose=0)
        exact_results = parallel(
            delayed(_get_evaluations)(
                perm=p,
                batch_creator=self.batch_creator,
                predictor=self.predictor,
                empty_value=self.empty_value,
                X=X,
            )
            for p in _get_permutations(self._ninstances)
        )

        for count, value in enumerate(exact_results):
            results[count, value[0]] = value[1]

        self.shapleyvalues_ = (1 / n) * np.sum(results, axis=0)
        return self.shapleyvalues_

    # def plot_values(
    #     self,
    #     nbest: Optional[int] = 10,
    #     names: Optional[Union[list, None]] = None,
    #     ax: Optional[Union[matplotlib.axes.Axes, None]] = None,
    # ) -> None:
    #     """
    #     Parameters
    #     ---------
    #     nbest: int, optional
    #         The default is 10
    #
    #     names: list, None, optional
    #         The default is None
    #
    #     ax : matplotlib.axes, optional
    #         The default is None.
    #
    #     Returns
    #     ------
    #     None
    #     """
    #     if ax is None:
    #         fig, ax = plt.subplots(figsize=(10, 6))
    #     if names is not None:
    #         df = pd.DataFrame(self.shapleyvalues_, index=names, columns=["shapley"])
    #     else:
    #         df = pd.DataFrame(
    #             self.shapleyvalues_,
    #             index=["instance_" + str(i) for i in range(len(self.shapleyvalues_))],
    #             columns=["shapley"],
    #         )
    #
    #     df["shapley_abs"] = np.abs(df["shapley"])
    #     df = df.sort_values(by="shapley_abs", ascending=False).iloc[
    #         : max(nbest, len(self.shapleyvalues_)), 0
    #     ]
    #
    #     df.plot.barh(color=(df > 0).map({True: "red", False: "blue"}), ax=ax)
    #     return


def _get_permutations(n: int) -> Generator[np.ndarray, None, None]:
    """Yields permutations of the natural numbers [0, ..., n-1]"""
    for p in permutations(range(n)):
        yield np.array(p)


def _get_evaluations(
    perm: np.ndarray,
    batch_creator: Callable,
    predictor,
    empty_value,
    X,
) -> Tuple[np.ndarray, np.ndarray]:
    """ """
    batch = batch_creator(perm, X)
    values = predictor(batch)
    marginals = _get_marginals(values, empty_value)
    return perm, marginals


def _get_marginals(v: np.ndarray, first_value: float) -> np.ndarray:
    """ """
    v_ = np.zeros(v.shape)
    v_[0] = first_value
    v_[1:] = v.copy()[0:-1]
    return v - v_
