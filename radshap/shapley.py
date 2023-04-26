import warnings
from typing import NoReturn, Optional, Tuple, Callable, Union, Generator
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


class Shapley:
    """Computes the Shapley value of every element of a collection of instances that are aggregated together in a single
     input of a trained predictive algoirhtm.
    It either uses an exact enumeration strategy when the number of samples is relatively small (<8) or an approximate
    Monte Carlo scheme with antithetic sampling.

    Parameters
    ----------
    predictor: Callable (input: 2D array of shape (n_inputs, n_input_features), output: 1D array of shape (n_inputs,))
        Trained predictor that returns one single real-value prediction per input.

    aggregator: Callable (input: 2D array of shape (n_instances, n_instance_features), output: 1D array of shape (n_input_features,))
        Aggregator that transforms an array of n_instances (with each one characterized by n_instance_features) into one
        single vector of shape (n_input_featurs) that will be used as an input of the predictor.

    estimation_method: str {'auto', 'exact', 'antithetic'}, optional
        Estimation method for the Shapley values. IfThe default is "auto".

    empty_value: float, otpional
        The default is 0.5.

    n_jobs: int, optional
        Number of jobs to run in parallel. -1 means using all processors.
        See the joblib package documentation for more explanations. The default is 1.

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
    >>> fun_agg: lambda x: np.mean(x, axis=0)
    >>> shap = Shaplay(predictor = lambda x: model.predict_proba(x)[:, 1], aggregator = fun_agg)
    >>> shap.explain(X) # X a 2D array of shape (n_instances, n_instance_features)
    """

    def __init__(
        self,
        predictor: Callable[[np.ndarray], float],
        aggregator: Callable[[np.ndarray], np.ndarray],
        empty_value: Optional[float] = 0.5,
        estimation_method: Optional[str] = "auto",
        nsamples: Optional[int] = 1000,
        n_jobs: Optional[int] = 1,
    ) -> NoReturn:

        self.nsamples = nsamples
        self.predictor = predictor
        self.empty_value = empty_value
        self.estimation_method = estimation_method
        self.n_jobs = n_jobs

        self.batch_creator = _BatchCreator(aggregator)
        self._ninstances = None
        self.shapleyvalues_ = None

    def explain(self, X: np.ndarray, **kwargs) -> object:
        """ Computes the Shapley values for each row of X.

        X must correspond to a valid collection of instances (shape (n_instances, n_instance_features)) that will be
        passed to the aggregator function and then used as an input to the predictor.

        Parameters
        ----------
        X: 2D array of shape (n_instances, n_instance_features)
            X corresponds to a collection of instances that will be aggregated into a single input

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._ninstances = X.shape[0]
        if self.estimation_method == "auto":
            if self._ninstances <= 7:
                return self._explain_exact(X, **kwargs)
            else:
                return self._explain_antithetic(X, **kwargs)
        elif self.estimation_method == "exact":
            if self._ninstances > 7:
                warnings.warn(
                    "The number of permutations to test for exact estimation is at least 40320."
                    " It may require some computational power."
                )
            return self._explain_exact(X, **kwargs)
        elif self.estimation_method == "antithetic":
            return self._explain_antithetic(X, **kwargs)
        else:
            raise ValueError("Unrecognized estimation method. Please choose among 'auto', 'exact', or antithetic'")

    def _explain_antithetic(self, X: np.ndarray, **kwargs) -> object:
        """

        """
        results = np.zeros((self.nsamples // 2, self._ninstances))
        parallel = Parallel(n_jobs=self.n_jobs, verbose=0)
        sampling_results = parallel(
            delayed(self._get_antithetic_evaluations)(X, **kwargs)
            for _ in range(self.nsamples // 2)
        )

        for count, value in enumerate(sampling_results):
            results[count, value[0]] = value[1]

        self.shapleyvalues_ = (1 / self.nsamples) * np.sum(results, axis=0)
        return self

    def _get_antithetic_evaluations(self, X: np.ndarray, **kwargs) -> object:
        """

        """
        p = np.random.permutation(np.arange(self._ninstances))
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

    def _explain_exact(self, X: np.ndarray, **kwargs) -> object:
        """

        """
        n = np.math.factorial(self._ninstances)
        results = np.zeros((n, self._ninstances))
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
            for p in _get_permutations(self._ninstances)
        )

        for count, value in enumerate(exact_results):
            results[count, value[0]] = value[1]

        self.shapleyvalues_ = (1 / n) * np.sum(results, axis=0)
        return self

    def plot_values(
            self,
            nbest: Optional[int] = 10,
            names: Optional[Union[list, None]] = None,
            ax: Optional[Union[matplotlib.axes.Axes, None]] = None
    ) -> None:
        """
        Parameters
        ---------
        nbest:

        names:

        ax : matplotlib.axes, optional
            The default is None.

        Returns
        ------
        None
        """
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


class _BatchCreator:
    def __init__(self, fun_agg):
        self.fun_agg = fun_agg

    def __call__(self, permutation, X, **kwargs):
        X_perm = np.copy(X)[permutation, :]
        L = [self.fun_agg(X_perm[:i, :], **kwargs).reshape(1, -1) for i in range(1, len(permutation) + 1)]
        return np.vstack(L)

        # n_instances = len(permutation)
        # X_perm_3d = np.tile(np.copy(X)[permutation, :], (n_instances, 1, 1))
        # mask_3d = np.tile(np.tril(np.ones((n_instances, n_instances)))[:, :, np.newaxis], (1, 1, X.shape[1]))


def _get_permutations(n: int) -> Generator[np.ndarray]:
    """

    """
    for p in permutations(range(n)):
        yield np.array(p)


def _get_evaluations(
        perm: np.ndarray,
        batch_creator: Callable,
        predictor,
        empty_value,
        X,
        **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """

    """
    batch = batch_creator(perm, X, **kwargs)
    values = predictor(batch)
    marginals = _get_marginals(values, empty_value)
    return perm, marginals


def _get_marginals(v: np.ndarray, first_value: float) -> np.ndarray:
    """

    """
    v_ = np.zeros(v.shape)
    v_[0] = first_value
    v_[1:] = v.copy()[0:-1]
    return v - v_
