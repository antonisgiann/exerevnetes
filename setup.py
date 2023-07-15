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
    description='Another auto machine learning implementation, the goal is to create a simple yet comperhensive library that automates model selection',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carnivore7777/exerevnetes",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "pandas",
        "skopt"
    ]
)

