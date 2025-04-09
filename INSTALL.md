# Install

This project has been built specifically for Python Version 3.9.0. Although not
required, if you want to ensure maximum compatibility with the code, please use
this version of Python. You can install Python 3.9.0 from the official website:
https://www.python.org/downloads/release/python-390/

## Setting up the Virtual Environment

A virtual environment is a self-contained directory that contains a Python 
installation for a particular version of Python, plus several additional
packages. It allows you to manage dependencies for different projects
separately.

To create a virtual environment, you can use the following command:

```bash
python3 -m venv env
```

To activate the virtual environment, use the following command:

```bash
source env/bin/activate
```

## Installing Required Packages

Once you have activated the virtual environment, you can install the required
packages using the following command:

```bash
pip install -r requirements.txt
```

This will install all the packages listed in the `requirements.txt` file, which
contains the dependencies for this project.

Now you are ready to run the project!