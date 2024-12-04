## Installing required packages
1. Install sphinx
```bash
    pip install sphinx
```
2. Install type hints for sphinx
```bash
    pip install sphinx-autodoc-typehints
```
3. Install myst-parser
```bash
    pip install myst-parser
```

## Generating the documentation

1. Method one:

Go to the directory where the file is located and run the following command:
```bash
    sphinx-build -b html "/home/cephadrius/Desktop/git/Lexi-BU/lexi/" "/home/cephadrius/Desktop/git/Lexi-BU/lexi/docs/"
```
This will generate the documentation in the docs folder.

2. Method two:
Go to the directory where the file is located and run the following command:
```bash
    make clean
    make html
```
`make clean` will remove the existing build files and `make html` will generate the documentation in
the build folder.

