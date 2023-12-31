## exerevnetes

`exerevnetes` is a python library with the goal to provide a simple and yet comperhensive framework for model and pipeline comparison with minimal dependencies.

**⚠️ Note: For a more full - capable library please see <a href=https://pycaret.org/>pycaret</a>.**

## Table of contents

- [Table of conents](#table-of-contents)
- [Requirements](#requirements)
- [Installation](#installation-work-in-progress)
- [Usage](#usage-work-in-progress)
- [Contributions](#contributions)
- [Local Setup - Development](#local-setup---development)

### Requirements

- The library expects a preprocessed dataset in the form of <b>X</b> (independent variables) and <b>y</b> (dependent variable).
- X and y should be split and be given to the constructor as seperate arguments.
- The only mandatory arguments for the comparator to run are <b>Χ</b> and <b>y</b>. All other arguments have predefined values.
- Every estimator used should be scikit-learn compatible and be able to be used in [cross val predict](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html).
- If the dataset is not preprocessed there is a <b>preprocess</b> argument that can be set to the user defined preprocessing.
- If an argument is used then it should not be empty (e.g. estimators, preprocess, exclude).
- The expected output after the execution of <b>run()</b> is some run time information about the algorithm that is being tested and training times and a pandas dataframe at the end showing all the metrics for each algorithm that was tested.

### Installation (work in progress)

<i>to be implemented</i>

### Usage (work in progress)
<h4>Classification</h4>
<ul>
    <li>Binary Classification</li>
   
    from exerevnetes import BinaryClassifierComparator # import

    cmp = BinaryClassifierComparator(X, y) # basic comparator initialization

    cmp.run() # run the comparator, which at the end displays the results

    results = cmp.get_results() # returns the results dataframe

    clf = cmp.get_best(metric="precision_score") # returns the classifier that achieved the higher metric
</ul>
<ul>
    <li>Multiclass classification (to be implemented)</li>
</ul>
<ul>
    <li>Regression</li>
    
    from exerevnetes import RegressionComparator # import

    cmp = RegressionComparator(X, y) # basic comparator initialization

    cmp.run() # run the comparator, which at the end displays the results

    results = cmp.get_results() # returns the results dataframe

    est = cmp.get_best(metric="mean_absolute_error")
</ul>

## Contributions

This is an open source project and I welcome any contributions - ideas

## Local Setup - Development

1. Clone the repository locally

<b>ssh</b>
```
git clone git@github.com:carnivore7777/exerevnetes.git
```
<b>https</b>
```
git clone https://github.com/carnivore7777/exerevnetes.git
```
2. Install poetry for the python environment

Head over to [Poetry official webpage](https://python-poetry.org/docs/) and follow the installation instructions

3. Run poetry to set upt the environment
```
poetry install
```
4. Activate the environment
```
poetry shell
```