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
## License
This project is licensed under a custom open-source license (see the [LICENSE.md](LICENSE.md) file for more details).
## Acknowledgements

This package was created as a part of the PhD project of Nicolas Captier in the [Laboratory of Translational Imaging in Oncology (LITO)](https://www.lito-web.fr/en/) of Institut Curie.