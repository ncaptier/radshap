import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="radshap",
    version="0.0.1a",
    author="Nicolas Captier",
    author_email="nicolas.captier@curie.fr",
    description="Shapley values for interpreting radiomic models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ncaptier/radshap",
    packages=setuptools.find_packages(),
    install_requires=[
        "matplotlib",
        "numpy",
        "joblib>=1.1.0",
        "pandas>=1.3.5",
        "seaborn>=0.11.2",
        "SimpleITK>=1.2.4"
    ],

    extras_require={"dev": ["pytest"],
                    "docs": ["sphinx == 7.1.2", "sphinx-gallery == 0.14.0", "numpydoc == 1.5.0", "nbsphinx == 0.9.3",
                             "ipython==8.12.2", "sphinx-rtd-theme==1.3.0"]},

    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.7',
)