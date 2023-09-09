Installation
============

There are two different ways to install **radshap** :

* :ref:`Install the latest official release with PyPI <install_pypi>`. It will provide you the latest stable version of **radshap** (recommended for new users).
* :ref:`Install the latest development version with GitHub <install_github>`. It will provide you the latest features but it may come with some instabilities.

.. _install_pypi:

Installing from PyPI
--------------------

	>>> pip install radshap


.. _install_github:

Installing from GitHub
----------------------

	>>> pip install git+https://github.com/ncaptier/radshap.git

Dependencies
------------

=================  =================
   Dependency       Minimum version
=================  =================
   joblib               1.1.0
   matplotlib
   numpy
   pandas               1.3.5
   scikit-learn         1.0.2
   seaborn              0.11.2
   SimpleITK            1.2.4
=================  =================

.. warning::

   **stabilized-ica** requires at least Python 3.7.

.. note::

   Both installing options should install all the dependencies.