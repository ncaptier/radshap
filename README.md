# radshap

<p align="center">
    <img src="docs/images/radshap_logo.png"/>
</p>

[![Documentation Status](https://readthedocs.org/projects/radshap/badge/?version=latest)](https://radshap.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://badge.fury.io/py/radshap.svg)](https://badge.fury.io/py/radshap)
[![Downloads](https://pepy.tech/badge/radshap)](https://pepy.tech/project/radhsap)

This repository proposes a python tool for highlighting the contribution of different regions of interest (ROIs) to the predictions of radiomic models.
It estimates the Shapley value of the different ROIs of an image that a trained radiomic model uses to obtain a prediction.

## Graphical abstract

<p align="center">
    <img src="docs/images/graphical_abstract.png"/>
</p>
<b>a.</b> schematic view of a generic aggregated radiomic model - <b>b.</b> computation of a Shapley value for a specific region.

## Documentation

[https://radshap.readthedocs.io/en/latest](https://radshap.readthedocs.io/en/latest)

## Install


### Install the latest stable version with PyPi
```
pip install radshap
```

### Install from source
```
pip install git+https://github.com/ncaptier/radshap.git
```

## Experiments
We provide a jupyter notebook for an illustration with PET images and simple aggregation strategies:
* [Classification of Non-Small Cell Lung Cancer subtype and interpretation with Shapley values](examples/nsclc_subtype_classification.ipynb)

We provide a jupyter notebook for an illustration with PET images and custom aggregation strategies:
* [PFS prediction for NSCLC patients undergoing immunotherapy and interpretation with Shapley values](examples/nsclc_survival_prediction.ipynb)

We provide a jupyter notebook for an illustration of a robust strategy for computing Shapley values:
* [Robust Shapley values for explaining multi-region radiomic models with non-optional regions](examples/robust_shapleyvalues.ipynb)

## Examples
**Explanation with Shapley values**
```python
import numpy as np
import joblib
from radshap.shapley import Shapley

model = joblib.load("trained_logistic_regression.joblib")
shap = Shapley(predictor = lambda x: model.predict_proba(x)[:, 1], aggregation = ('mean', None))
shapvalues = shap.explain(X) # X a 2D array of shape (n_instances, n_instance_features)
```

**Robust explanation with Shapley values**
```python
import numpy as np
import joblib
from radshap.shapley import RobustShapley

model = joblib.load("trained_logistic_regression.joblib")
shap = RobustShapley(predictor = lambda x: model.predict_proba(x)[:, 1],
                     aggregation = ('nanmean', None),
                     background_data = Xback) # Xback a 2D array of shape (n_samples_background, n_input_features)
shapvalues = shap.explain(X) # X a 2D array of shape (n_instances, n_instance_features)
```

**Explanation with Shapley values and custom aggregation function**
```python
import numpy as np
import joblib
from radshap.shapley import Shapley

model = joblib.load("trained_linear_regression.joblib")
# Compute the average prediction to approximate a "random" prediction with no information (required for RadShap)
predictions = np.load('predictions.npy')
mean_pred = predictions.mean()

def custom_agg_function(Xsub):
    """ Aggregate an arbitrary subset of regions (Xsub array with and arbitray 
    number of rows) into a valid aggregated input for the predictive model.
    
    Parameters
    ---------
    Xsub: 2D array of shape (n_instances, n_instance_features)
    
    Returns
    -------
    agg_input: 1D array of shape (1, n_input_features)
    """ 
    
    ... #aggregate information from the differente regions in Xsub (i.e rows)
    ... #to obtain a valid aggregated input for the predictive model
    
    return agg_input

shap = Shapley(predictor = lambda x: model.predict(x),
               aggregation = custom_agg_function,
               empty_value = mean_pred)
shapvalues = shap.explain(X) # X a 2D array of shape (n_instances, n_instance_features)
```

## Citing RadShap

If you use RadShap in a scientific publication, please cite the [following paper](https://jnm.snmjournals.org/content/early/2024/06/21/jnumed.124.267434):

```
Nicolas Captier, Fanny Orlhac, Narinée Hovhannisyan-Baghdasarian, Marie Luporsi, Nicolas Girard and Irène Buvat. RadShap: An Explanation Tool for Highlighting the Contributions of Multiple Regions of Interest to the Prediction of Radiomic Models. Journal of Nuclear Medicine, 2024, DOI: https://doi.org/10.2967/jnumed.124.267434 
```

## License
This project is licensed under a custom open-source license (see the [LICENSE.md](LICENSE.md) file for more details).
## Acknowledgements

This package was created as a part of the PhD project of Nicolas Captier in the [Laboratory of Translational Imaging in Oncology (LITO)](https://www.lito-web.fr/en/) of Institut Curie.