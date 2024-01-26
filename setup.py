from setuptools import setup, find_packages

setup(
    name='lexi",
    version='0.1',
    description="Data analysis tools for the Lexi project",
    long_description=="open('README.md').read()",
    url="https://github.com/Lexi-BU/lexi",
    author='Lexi',
    author_email="lunar.lexi01@gmail.com",
    license="MIT",
    keywords="data analysis",
    packages=find_packages(),
    install_requires=["spacepy"],
    python_requires=">=3.10",
)
