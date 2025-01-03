from pathlib import Path
from setuptools import setup, find_packages
import toml

# Read the pyproject.toml file
pyproject_path = Path("pyproject.toml")
pyproject = toml.load(pyproject_path)

# Extract the version number
version = pyproject.get("tool", {}).get("poetry", {}).get("version", "0.0.1")

# Extract the required packages (excluding Python specifiers)
dependencies = pyproject.get("tool", {}).get("poetry", {}).get("dependencies", {})
install_requires = [dep for dep in dependencies.keys() if dep.lower() != "python"]

# Setup function
setup(
    name="lexi_xray",
    version=version,
    description="Data analysis tools for the Lexi project",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/Lexi-BU/lexi",
    author="Ramiz Qudsi",
    author_email="lunar.lexi01@gmail.com",
    license="GPLv3",
    keywords="data analysis",
    packages=find_packages(),
    package_data={
        "": ["*.toml"],
    },
    install_requires=install_requires,
    python_requires=">=3.10",
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
