import toml
from setuptools import setup, find_packages

# Read the pyproject.toml file
with open("pyproject.toml") as f:
    pyproject = toml.load(f)

# Extract the version number
version = pyproject["tool"]["poetry"]["version"]

# Extract the required packages
install_requires = pyproject["tool"]["poetry"]["dependencies"].keys()

setup(
    name="lexi",
    version=version,
    description="Data analysis tools for the Lexi project",
    long_description=open("README.md").read(),
    url="https://github.com/Lexi-BU/lexi",
    author="Lexi",
    author_email="lunar.lexi01@gmail.com",
    license="MIT",
    keywords="data analysis",
    packages=find_packages(),
    package_data={
        "": ["*.toml"],
    },
    install_requires=install_requires,
    python_requires=">=3.10",
    include_package_data=True,
)
