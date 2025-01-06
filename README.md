<div align="center">
    <img src="https://raw.githubusercontent.com/Lexi-BU/lexi/stable/images/lexi_logo.png" alt="LEXI Logo" width="200" height="131">
</div>


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14582916.svg)](https://doi.org/10.5281/zenodo.14582916)

A python package for data analysis related to [LEXI](https://lexi-bu.github.io/).

# Installation Guide

The next section of this document will guide you through the installation process of `lexi`.

Though it is not necessary, we strongly recommend that you install `lexi` in a virtual environment.
This will prevent any conflicts with other Python packages you may have installed.

A virtual environment is a self-contained directory tree that contains a Python installation for a
particular version of Python, plus a number of additional packages. You can install packages into a
virtual environment without affecting the system's Python installation. This is especially useful when
you need to install packages that might conflict with other packages you have installed.

## Creating a virtual environment
There are several ways to create a virtual environment. We recommend using `python3` to do so.

For this exercise, we will assume that you have a directory called `Documents/lexi` where you will
install `lexi` and create your virtual environment. Please replace `Documents/lexi` with the actual
path to the directory where you want to install `lexi` and create your virtual environment.

- cd into `Documents/lexi`

### Using python3
You can create a virtual environment called `lexi_venv` (or any other name you might like) using 
`python3` by running the following command:

```bash
    python3 -m venv lexi_venv
```

You can activate the virtual environment by running the following command:

#### on Linux/MacOS:

```bash
    source lexi_venv/bin/activate
```

#### on Windows:

```bash
    .\lexi_venv\Scripts\activate
```

You can deactivate the virtual environment by running the following command:

```bash
    deactivate
```

## Installing `lexi`

### Installing from PyPI
After you have created and activated your virtual environment, you can install `lexi` from PyPI by running the following command:

```bash
    pip install lexi_xray
```

### Installing from source
After you have created and activated your virtual environment, you can install `lexi` directly from
GitHub by running the following command:

```bash
    pip install git+https://github.com/Lexi-BU/lexi
```
NOTE: This will install the latest version of `lexi` from the main branch. If you want to install a
specific version, please append the version number to the URL.
For example, if you want to install version `0.3.1`, you can run the following command:

```bash
    pip install git+https://github.com/Lexi-BU/lexi@0.3.1
```

## Verifying the installation
You can verify that `lexi` was installed by running the following command:

```bash
    pip show lexi_xray
```

which should produce output similar to the following:

```
    Name: lexi_xray
    Version: 0.0.1
    Summary: Main repository for all data analysis related to LEXI
    Home-page: 
    Author: qudsiramiz
    Author-email: qudsiramiz@gmail.com
    License: GNU GPLv3
    Location: /home/cephadrius/Desktop/lexi_code_test_v2/lexi_test_v2/lib/python3.10/site-packages
    Requires: cdflib, matplotlib, pandas, pytest, toml
    Required-by: 
```

You can also verify that `lexi` was installed by running the following command:

```bash
    pip list
```
which should produce output similar to the following:

```bash
    Package         Version
    --------------- -------
    .....................
    kiwisolver      1.4.5
    lexi_xray         0.4.1
    matplotlib      3.8.2
    numpy           1.26.4
    .....................
```

You can open a Python shell and import `lexi` by running the following command:

```bash
    python
    from lexi_xray import lexi as lexi
    import lexi_xray
    lexi_xray.__version__
``` 

which should produce output similar to the following:

```bash
'0.4.1'
```
If that worked, congratulations! You have successfully installed `lexi`.


# Using LEXI Software

NOTE: We will add more examples and tutorials in the future. For now, we will use a Jupyter Notebook
to demonstrate how to use `lexi` to analyze data from LEXI.

## Using the Example Google Colab Notebook
1. 1. If you haven't already, download the example notebook from the following link:
    [Concise
      Tutorial](https://colab.research.google.com/drive/1Q0dmH7QrwRXZh8ZrzfOQbshBA-B86y6T?usp=sharing)

    [Detailed Tutorial](https://colab.research.google.com/drive/1rVOE_INV3bO2O_s0K7u58zHxbNhawELt?usp=sharing)
2. Open the notebook in Google Colab by clicking on the link above.

3. The notebook will then guide you through the process of using `lexi` to analyze data from LEXI.

4. If you want to run the notebook on your local machine, you can download the notebook from the link
   above and run it in a Jupyter Notebook environment.

5. If you encounter any issues, please report them to us by creating an issue on our GitHub
   repository [here](https://github.com/Lexi-BU/lexi/issues).

## Citation
If you use `lexi` in your research, please cite the following paper:

```
    @Software{Qudsi2025,
        author    = {Qudsi, Ramiz and Chitty, Zoe and Connor, Cadin and Walsh, Brian},
        title     = {Lexi-BU/lexi: v0.4.0},
        doi       = {10.5281/zenodo.14606885},
        url       = {https://doi.org/10.5281/zenodo.14606885},
        version   = {v0.4.0},
        month     = jan,
        publisher = {Zenodo},
        path=Lexi-BU-lexi-e01a2a4 },
        year      = {2025},
    }
```
