from typing import NoReturn, Optional, Callable, Union

import numpy as np


class IndependentMaskedModel:
    """ Return a new model that masks out features associated with invalid inputs by integrating over a given
     background data set.

    Parameters
    ----------
    model: callable (input: 2D array of shape (n_inputs, n_input_features), output: 1D array of shape (n_inputs,))
        Original model whose invalid inputs need to be handled.

    background_data: 2D array of shape (n_background, n_input_features)
        Background data set used to deal with invalid inputs for the model. In the case of an invalid input, we
        generate "n_background" new inputs by replacing the invalid values with corresponding values from the background
        data set. We then apply the model to each of these generated inputs and return the average output.

    invalid_features: 1D array of shape (n_input_features)
        Boolean array to specify the features for which a NaN value means that the input is not valid. In such cases, we
        handle them with the background data set. If None, all the features will be taken into account to define invalid
        inputs.
    """

    def __init__(
            self,
            model: Callable[[np.ndarray], float],
            background_data: np.ndarray,
            invalid_features: Union[np.ndarray, None]
    ) -> NoReturn:

        self.model = model
        self.background_data = background_data
        if masked_features is None:
            self.invalid_features = np.full(background_data.shape[1], True)
        else:
            self.invalid_features = invalid_features

    def __call__(self, X: np.ndarray) -> np.ndarray:
        nsamples = X.shape[0]
        results = np.zeros(nsamples)
        for i in range(nsamples):
            bool_mask = self.invalid_features * np.isnan(X[i, :])
            if np.any(bool_mask):
                Xnew = np.repeat(X[i, :].reshape(1, -1), repeats=self.background_data.shape[0], axis=0)
                Xnew[:, bool_mask] = self.background_data[:, bool_mask]
                results[i] = np.mean(self.model(Xnew))
            else:
                results[i] = self.model(X[i, :].reshape(1, -1))
        return results
