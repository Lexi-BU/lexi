Installation Guide
==================

The next section of this document will guide you through the installation process of `lexi`.

Though it is not necessary, we strongly recommend that you install Lexi in a virtual environment.
This will prevent any conflicts with other Python packages you may have installed.

A virtual environment is a self-contained directory tree that contains a Python installation for a
particular version of Python, plus a number of additional packages. You can install packages into a
virtual environment without affecting the system Python installation. This is especially useful when
you need to install packages that might conflict with other packages you have installed.

Creating a Virtual Environment
------------------------------

There are several ways to create a virtual environment. We recommend using `python3` to do so.

For this exercise, we will assume that you have a directory called `Documents/lexi` where you will
install `lexi` and create your virtual environment. Please replace `Documents/lexi` with the actual
path to the directory where you want to install `lexi` and create your virtual environment.

- Navigate to the `Documents/lexi` directory.

Using python3
~~~~~~~~~~~~~

You can create a virtual environment called `lexi_venv` (or any other name you might like) using 
`python3` by running the following command:

.. code-block:: bash

    python3 -m venv lexi_venv

You can activate the virtual environment by running the following command:

On Linux/MacOS:
^^^^^^^^^^^^^^^

.. code-block:: bash

    source lexi_venv/bin/activate

On Windows:
^^^^^^^^^^^

.. code-block:: bash

    .\lexi_venv\Scripts\activate

You can deactivate the virtual environment by running the following command:

.. code-block:: bash

    deactivate

Installing `lexi`
-----------------
There are three ways to install `lexi`: from pypi, from source, and from a local copy.

Installing from PyPI
~~~~~~~~~~~~~~~~~~~~

After you have created and activated your virtual environment, you can install `lexi` from PyPI by
running the following command:

.. code-block:: bash

    pip install lexi_xray


Installing from Source
~~~~~~~~~~~~~~~~~~~~~~

After you have created and activated your virtual environment, you can install `lexi` directly from
GitHub by running the following command:

.. code-block:: bash

    pip install git+https://github.com/Lexi-BU/lexi

.. note::
    This will install the latest version of `lexi` from the main branch. If you want to install a specific version, please append the version number to the URL.
    For example, if you want to install version `0.3.1`, you can run the following command:

    .. code-block:: bash

        pip install git+https://github.com/Lexi-BU/lexi@0.3.1



Verifying the Installation
==========================

You can verify that `lexi` was installed by running the following command:

.. code-block:: bash

    pip show lexi_xray

This should produce output similar to the following:

.. code-block::

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

You can also verify that `lexi` was installed by running the following command:

.. code-block:: bash

    pip list

This should produce output similar to the following:

.. code-block:: bash

    Package         Version
    --------------- -------
    .....................
    kiwisolver      1.4.5
    lexi_xray         0.4.1
    matplotlib      3.8.2
    numpy           1.26.4
    .....................

You can open a Python shell and import `lexi` by running the following commands:

.. code-block:: bash

    python
    from lexi_xray import lexi as lexi
    import lexi_xray
    lexi_xray.__version__

This should produce output similar to the following:

.. code-block::

    '0.4.1'

If that worked, congratulations! You have successfully installed `lexi`.

Using `lexi` Software
=====================

.. note::
   We will add more examples and tutorials in the future. For now, we will use a Google Colab Notebook
   to demonstrate how to use `lexi` to analyze data from LEXI.

Using the Example Google Colab Notebook
----------------------------------------

1. If you haven't already, download the example notebook from the following link: 
   `Concise Tutorial
   <https://colab.research.google.com/drive/1Q0dmH7QrwRXZh8ZrzfOQbshBA-B86y6T?usp=sharing>`_

   `Detailed Tutorial <https://colab.research.google.com/drive/1rVOE_INV3bO2O_s0K7u58zHxbNhawELt?usp=sharing>`_

2. Open the notebook in Google Colab by clicking on the link above.

3. The notebook will then guide you through the process of using `lexi` to analyze data from LEXI.

4. If you want to run the notebook on your local machine, you can download the notebook from the link
   above and run it in a Jupyter Notebook environment.

5. If you encounter any issues, please report them to us by creating an issue on our GitHub
   repository `here <https://github.com/Lexi-BU/lexi/issues>`_.