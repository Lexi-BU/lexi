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
    pip install lexi_bu
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

### Installing from a local copy
We don't recommend this, however if you must, here are the details.

After you have created and activated your virtual environment, you can install `lexi` from a local copy
by following these steps:

NOTE: We will use version `0.3.4` as an example. Please replace `0.3.4` with the actual version number
    you want to install.

1. Download `lexi-0.3.4.tar.gz` directory from the following link: [Download LEXI
   Software](https://github.com/Lexi-BU/lexi/archive/refs/tags/v0.3.4.tar.gz)

2. Copy the `lexi-0.3.4.tar.gz` file into `Documents/lexi` (or any other directory where you want
   to install `lexi` in).

3. Activate your virtual environment using the instructions above.

4. Install `lexi` by running the following command (NOTE: replace `lexi-0.3.4.tar.gz` with the actual
   name of the file you downloaded):

    ```bash
        pip install lexi-0.3.4.tar.gz
    ```

This will install `lexi` and all its dependencies.

## Verifying the installation
You can verify that `lexi` was installed by running the following command:

```bash
    pip show lexi_bu
```

which should produce output similar to the following:

```
    Name: lexi_bu
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
    lexi_bu         0.0.1
    matplotlib      3.8.2
    numpy           1.26.4
    .....................
```

You can open a Python shell and import `lexi` by running the following command:

```bash
    python
    import lexi_bu as lexi
    lexi.__version__
``` 

which should produce output similar to the following:

```bash
'0.0.1'
```
If that worked, congratulations! You have successfully installed `lexi`.


# Using LEXI Software

NOTE: We will add more examples and tutorials in the future. For now, we will use a Jupyter Notebook
to demonstrate how to use `lexi` to analyze data from LEXI.

## Using the Example Jupyter Notebook
1. If you haven't already, download the example folder from the following link: [Download LEXI
Examples](https://lexi-bu.github.io/software/examples.zip) and extract it to a directory of your
choice. We will refer to this directory as `examples` for the rest of this document.

2. Activate your virtual environment. If you haven't already created a virtual environment, please
   refer to the [creating a virtual environment](#creating-a-virtual-environment) section for
   instructions on how to do so. Remember that you can activate your virtual environment by running
   the following command:

#### on Linux/MacOS:

```bash
    source lexi_venv/bin/activate
```

#### on Windows:

```bash
    .\lexi_venv\Scripts\activate
```

3. `cd` into the `examples` directory and 

4. If you haven't already, install Jupyter Notebook by running the following command:

```bash
    pip install jupyter
```

5. Open the Jupyter Notebook by running the following command:

```bash
    jupyter notebook lexi_tutorial.ipynb
```

6. This will open a new tab in your web browser and will look like the image below:
<div align="center">
    <img src="https://raw.githubusercontent.com/Lexi-BU/lexi/refs/heads/stable/images/lexi_notebook_screeenshot.png" alt="Jupyter Notebook">
</div>

7. You can now run the cells in the Jupyter Notebook to see how to use `lexi` to analyze data from
   LEXI.


## Citation
If you use `lexi` in your research, please cite the following paper:

```
    @software{ramiz_qudsi_2025_14585868,
                author       = {Ramiz Qudsi and
                                Zoe Chitty and
                                Cadin Connor and
                                Brian Walsh},
                  title        = {Lexi-BU/lexi: v0.3.4},
                  month        = jan,
                  year         = 2025,
                  publisher    = {Zenodo},
                  version      = {v0.3.4},
                  doi          = {10.5281/zenodo.14585868},
                  url          = {https://doi.org/10.5281/zenodo.14585868},
                  swhid        = {swh:1:dir:0f71ede7ba2db5f1548c45f330b2c945af59794b
                                   ;origin=https://doi.org/10.5281/zenodo.14582915;vi
                                   sit=swh:1:snp:6552c7107029bb957e1e2592742d13109209
                                   1a11;anchor=swh:1:rel:5ca1c846400ca145a0d2599c9951
                                   04421046558a;path=Lexi-BU-lexi-74dd3c2
                                  },
}
```
