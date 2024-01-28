This documents guides you through the installation of Lexi.

Though it is not necessary, we strongly recommend that you install Lexi in a virtual environment.
This will prevent any conflicts with other Python packages you may have installed.

## Creating a virtual environment
There are several ways to create a virtual environment. We recommend using `python3` or `poetry`.

For this exercise, we will assume that you have a directory called `Documents/lexi` where you will
install Lexi and create your virtual environment.

- cd into `Documents/lexi`
    - `cd Documents/lexi`
### Using python3
    - You can create a virtual environment called `lexi_venv` using `python3` by running the following command:
        - `python3 -m venv lexi_venv`
    - You can activate the virtual environment by running the following command:
        - on Linux/MacOS:
            - `source lexi_venv/bin/activate`
        - on Windows:
            - `.\lexi_venv\Scripts\activate.bat`
    - You can deactivate the virtual environment by running the following command:
        - `deactivate`
### Using poetry
    - You can create a virtual environment called `lexi_venv` using `poetry` by running the following command:
        - `poetry init`
    - You can activate the virtual environment by running the following command:
        - `poetry shell`
    - You can deactivate the virtual environment by running the following command:
        - `exit`

## Installing Lexi
    - Copy the `lexi/dist` directory into `Documents/lexi`.
    - NOTE: Since we do not have proper sky background data and the ephemeris data, we have to use data locally available. You thus must have the following files in the `Documents/lexi` directory:
        - `data/PIT_shifted_jul08.cdf`
        - `data/sample_lexi_pointing_ephem_edited.csv`
        - `data/sample_xray_background.csv`

    - The `lexi/dist` directory contains a file called `lexi-0.0.1.tar.gz`, or some other version of the same file.
    - Activate your virtual environment uusing the instructions above.
    - Install Lexi by running the following command:
        - `pip install dist/lexi-0.0.1.tar.gz`
    - You can verify that Lexi was installed by running the following command:
        - `pip show lexi` which should produce output similar to the following:
            ```
            Name: lexi
            Version: 0.0.1
            Summary: Main repository for all data analysis related to LEXI
            Home-page: 
            Author: qudsiramiz
            Author-email: qudsiramiz@gmail.com
            License: GNU GPLv3
            Location: /home/vetinari/Desktop/lxi_code_test/lxi_code_testv0/lib/python3.10/site-packages
            Requires: pandas, spacepy, toml
            Required-by: 
            ```
    - You can also verify that Lexi was installed by running the following command:
        - `pip list` which should produce output similar to the following:
            ```
            Package         Version
            --------------- -------
            .....................
            lexi            0.0.1
            pandas          1.3.4
            pip             21.3.1
            .....................
            ```
    - You can open a Python shell and import Lexi by running the following command:
        - `python`
        - `import lexi`
        - `lexi.__version__` which should produce output similar to the following:
            ```
            '0.0.1'
            ```
    - If that worked, congratulations! You have successfully installed Lexi.