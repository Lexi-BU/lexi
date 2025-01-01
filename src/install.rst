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
install Lexi and create your virtual environment. Please replace `Documents/lexi` with the actual
path to the directory where you want to install Lexi and create your virtual environment.

- Change into the `Documents/lexi` directory.

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

Installing Lexi
---------------
There are three ways to install Lexi: from pypi, from source, and from a local copy.

Installing from PyPI
~~~~~~~~~~~~~~~~~~~~~

After you have created and activated your virtual environment, you can install Lexi from PyPI by
running the following command:

.. code-block:: bash

    pip install lexi_bu


Installing from Source
~~~~~~~~~~~~~~~~~~~~~~

After you have created and activated your virtual environment, you can install Lexi directly from GitHub by running the following command:

.. code-block:: bash

    pip install git+https://github.com/Lexi-BU/lexi

.. note::
    This will install the latest version of `lexi` from the main branch. If you want to install a specific version, please append the version number to the URL.
    For example, if you want to install version `0.3.1`, you can run the following command:

    .. code-block:: bash

        pip install git+https://github.com/Lexi-BU/lexi@0.3.1


Installing from a Local Copy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After you have created and activated your virtual environment, you can install Lexi from a local copy
by following these steps:

1. Download `lexi-version.tar.gz` directory from the following link: `Download LEXI Software <https://lexi-bu.github.io/software/dist/lexi-0.0.1.tar.gz>`_

2. Copy the `lexi-version.tar.gz` file into `Documents/lexi` (or any other directory where you want
   to install lexi in).

3. Activate your virtual environment using the instructions above.

4. Install Lexi by running the following command (replace ``lexi-version.tar.gz`` with the actual name of the file you downloaded):

.. code-block:: bash

    pip install lexi-version.tar.gz

This will install Lexi and all its dependencies.

Verifying the Installation
==========================

You can verify that Lexi was installed by running the following command:

.. code-block:: bash

    pip show lexi_bu

This should produce output similar to the following:

.. code-block::

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

You can also verify that Lexi was installed by running the following command:

.. code-block:: bash

    pip list

This should produce output similar to the following:

.. code-block:: bash

    Package         Version
    --------------- -------
    .....................
    kiwisolver      1.4.5
    lexi_bu         0.0.1
    matplotlib      3.8.2
    numpy           1.26.4
    .....................

You can open a Python shell and import Lexi by running the following commands:

.. code-block:: bash

    python
    import lexi_bu as lexi
    lexi.__version__

This should produce output similar to the following:

.. code-block::

    '0.0.1'

If that worked, congratulations! You have successfully installed Lexi.

Using LEXI Software
===================

.. note::
   We will add more examples and tutorials in the future. For now, we will use a Jupyter Notebook
   to demonstrate how to use `lexi` to analyze data from LEXI.

Using the Example Jupyter Notebook
----------------------------------

1. If you haven't already, download the example folder from the following link: 
   `Download LEXI Examples <https://lexi-bu.github.io/software/examples.zip>`_ 
   and extract it to a directory of your choice. We will refer to this directory as `examples` for the rest of this document.

2. Activate your virtual environment. If you haven't already created a virtual environment, please
   refer to the :ref:`creating-a-virtual-environment` section for instructions on how to do so.
   Remember that you can activate your virtual environment by running the following command:

On Linux/MacOS:
^^^^^^^^^^^^^^^

.. code-block:: bash

    source lexi_venv/bin/activate

On Windows:
^^^^^^^^^^^

.. code-block:: bash

    .\lexi_venv\Scripts\activate

3. Change into the `examples` directory.

4. If you haven't already, install Jupyter Notebook by running the following command:

.. code-block:: bash

    pip install jupyter

5. Open the Jupyter Notebook by running the following command:

.. code-block:: bash

    jupyter notebook lexi_tutorial.ipynb

6. This will open a new tab in your web browser and will look like the image below:

.. image:: _static/lexi_notebook_screeenshot.png
   :alt: Jupyter Notebook Screenshot

7. You can now run the cells in the Jupyter Notebook to see how to use `lexi` to analyze data from
   LEXI.
