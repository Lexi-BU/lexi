# lexi
Main repository for all data analysis related to LEXI

This document guides you through the installation of Lexi.

Though it is not necessary, we strongly recommend that you install Lexi in a virtual environment.
This will prevent any conflicts with other Python packages you may have installed.

A virtual environment is a self-contained directory tree that contains a Python installation for a
particular version of Python, plus a number of additional packages. You can install packages into a
virtual environment without affecting the system Python installation. This is especially useful when
you need to install packages that might conflict with other packages you have installed.

## Creating a virtual environment
There are several ways to create a virtual environment. We recommend using `python3` to do so.

For this exercise, we will assume that you have a directory called `Documents/lexi` where you will
install Lexi and create your virtual environment.

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
    .\lexi_venv\Scripts\activate.bat
```

You can deactivate the virtual environment by running the following command:

```bash
    deactivate
```

## Installing Lexi

### Installing from source
After you have created and activated your virtual environment, you can install Lexi directly from GitHub by running the following command:

```bash
    pip install git+https://github.com/Lexi-BU/lexi
```

### Installing from a local copy
After you have created and activated your virtual environment, you can install Lexi from a local copy
by following these steps:

1. Download `lexi-version.tar.gz` directory from the following link: [Download LEXI Software](https://lexi-bu.github.io/software/dist/lexi-0.0.1.tar.gz)

2. Copy the `lexi-version.tar.gz` file into `Documents/lexi`

3. Activate your virtual environment using the instructions above.

4. Install Lexi by running the following command (NOTE: replace `lexi-version.tar.gz` with the actual name of the file you downloaded):

    ```bash
        pip install lexi-version.tar.gz
    ```

This will install Lexi and all its dependencies.

## Verifying the installation
You can verify that Lexi was installed by running the following command:

```bash
    pip show lexi
```

which should produce output similar to the following:

```
    Name: lexi
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

You can also verify that Lexi was installed by running the following command:

```bash
    pip list
```
which should produce output similar to the following:

```bash
    Package         Version
    --------------- -------
    .....................
    kiwisolver      1.4.5
    lexi            0.0.1
    matplotlib      3.8.2
    numpy           1.26.4
    .....................
```

You can open a Python shell and import Lexi by running the following command:

```bash
    python
    import lexi
    lexi.__version__
``` 

which should produce output similar to the following:

```bash
'0.0.1'
```
If that worked, congratulations! You have successfully installed Lexi.
