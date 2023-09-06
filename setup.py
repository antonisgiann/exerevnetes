from setuptools import setup, find_packages

VERSIONFILE = "./exerevnetes/_version.py"
versionstr = open(VERSIONFILE, "rt").read()

if versionstr:
    version = versionstr.split()[-1].strip('"')
else:
    raise RuntimeError(f"Unable to find version in {VERSIONFILE}.")

with open("README.md", "rb") as fh:
    long_description = fh.read().decode('utf-8', errors='ignore')

setup(
    name='exerevnetes',
    version=version,
    author='Antonios Giannoulopoulos',
    author_email="antgiannoulopoulos@gmail.com",
    description='A machine learning algorithm comparator, the goal is to create a simple yet comperhensive that helps with model selection',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carnivore7777/exerevnetes",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "pandas",
        "catboost",
        "xgboost"
    ],
    classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    ]
)

