from typing import NoReturn, Optional, Tuple, Callable, Union
import numpy as np

_LIST_NUMPY = [
    "min",
    "max",
    "mean",
    "prod",
    "std",
    "sum",
    "var",
    "nanmin",
    "nanmax",
    "nanmean",
    "nanprod",
    "nanstd",
    "nansum",
    "nanvar",
]


def get_batch_creator(aggregation: Union[tuple, list, Callable]) -> Callable:
    """Build a batch creator that will aggregate several instances in a cumulative way.

    For a set of instances | instance_1 | the batch creator will output |            aggregator(instance_1)       |
                           | instance_2 |                               |     aggregator(instance_1, instance_2)  |
                                ...                                                            ...
                           | instance_k |                               | aggregator(instance_1, ..., instance_k) |
    Parameters
    ----------
    aggregation: tuple, list or tuple or callable
        Aggregation function.

        To define the aggregation funcion one can use::

            * a callable that takes as input a 2D array of shape (n_instances, n_instance_features) and returns a 1D
            array of shape (1, n_input_features).
            * a tuple (`method`, `subset`) with `method` being a string that refers to a numpy aggregating function (e.g
            'sum', 'min', 'std'...) and `subset` being a 1D array that defines the subset of columns/features on which
            to apply this method (or None for applying it on all the columns/features).
            * a list of tuples [(`method_1`, `subset_1`), (`method_2`, `subset_2`), ...] to define several aggregators.
            Please note that the aggregated features will be ordered according to the order of the provided list (i.e
            [agg_feature_method_1, ..., agg_feature_method_2, ...]).

    Returns
    -------
    callable
        Batch creator associated with the aggregation function

    Note
    ----
    If the aggregation function is defined with numpy aggregating functions (i.e. tuple or list of tuples) the batch
    creator will take advantage of numpy operations to speed up computations. Otherwise, a slow for loop will be used.
    """
    if callable(aggregation):
        return _NaiveBatchCreator(aggregation)
    elif isinstance(aggregation, (list, tuple)):
        return _FastBatchCreator(aggregation)
    raise TypeError(
        "aggregation should either be a callable (custom function), a tuple of the form (method (str),"
        "subset (None or np.ndarray)), or a list of such tuples"
    )


class _BatchCreator:
    def __init__(self, aggregation):
        self.fun_agg = _get_aggregation_function(aggregation)

    def check_fun_agg(self, X):
        pass

    def _create(self, permutation: np.ndarray, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, permutation: np.ndarray, X: np.ndarray) -> np.ndarray:
        return self._create(permutation, X)


class _NaiveBatchCreator(_BatchCreator):
    """Naive Batch creator for generic aggregation functions"""

    def check_fun_agg(self, X):
        try:
            test_results = self.fun_agg(X)
        except Exception as err:
            print(
                "Your custom aggregator is not properly defined."
            )
            print(f"The following {err}, {type(err)} occured")
            raise
        else:
            if (not isinstance(test_results, np.ndarray)) or (
                len(np.squeeze(test_results).shape) > 1
            ):
                print(
                    "Your custom aggregator should take as input a 2D array of shape "
                    "(n_instances, n_instance_features) and it should return a vector of shape (1, n_input_features)"
                    " (or (n_input_features,))."
                )
                raise

    def _create(self, permutation: np.ndarray, X: np.ndarray) -> np.ndarray:

        X_perm = np.copy(X)[permutation, :]
        L = [
            self.fun_agg(X_perm[:i, :]).reshape(1, -1)
            for i in range(1, len(permutation) + 1)
        ]
        return np.vstack(L)


class _FastBatchCreator(_BatchCreator):
    """Fast Batch creator for aggregation functions defined with numpy operations"""

    def _create(self, permutation: np.ndarray, X: np.ndarray) -> np.ndarray:
        n_instances = len(permutation)
        X_perm_3d = np.tile(np.copy(X)[permutation, :], (n_instances, 1, 1))
        mask_3d = np.tile(
            np.triu(np.ones((n_instances, n_instances)), k=1)[:, :, np.newaxis],
            (1, 1, X.shape[1]),
        )
        masked_array = np.ma.array(data=X_perm_3d, mask=mask_3d)
        return self.fun_agg(masked_array)


class _Aggregator3d:
    def __init__(self, method, subset):
        self.method = method
        self.subset = subset

    def __call__(self, masked_3d_array: np.ma.MaskedArray) -> np.ndarray:
        if self.subset is not None:
            return getattr(np, self.method)(
                masked_3d_array[:, :, self.subset], axis=1
            ).data
        return getattr(np, self.method)(masked_3d_array, axis=1).data


def _get_aggregation_function(aggregation: Union[tuple, list, Callable]) -> Callable:
    if callable(aggregation):
        return aggregation
    elif isinstance(aggregation, tuple):
        method, subset = _check_aggregation(aggregation)
        return _Aggregator3d(method=method, subset=subset)
    elif isinstance(aggregation, list):
        aggregation = [_check_aggregation(a) for a in aggregation]
        return lambda m: np.hstack(
            [_Aggregator3d(method=a[0], subset=a[1])(m) for a in aggregation]
        )
    raise TypeError(
        "aggregation should either be a callable (custom function), a tuple of the form (method (str),"
        "subset (None or np.ndarray)), or a list of such tuples"
    )


def _check_aggregation(tuple_aggregation: tuple) -> tuple:
    if (
        (len(tuple_aggregation) != 2)
        or (not isinstance(tuple_aggregation[0], str))
        or (
            not (
                (tuple_aggregation[1] is None)
                or (isinstance(tuple_aggregation[1], (list, np.ndarray)))
            )
        )
    ):
        raise ValueError(
            "aggregation only accepts tuples of the form (method (str), subset (None or np.ndarray))"
        )
    if not tuple_aggregation[0] in _LIST_NUMPY:
        raise ValueError("")
    return tuple_aggregation
