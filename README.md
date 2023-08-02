## exerevnetes

`exerevnetes` is a python library with the goal to provide a simple and yet comperhensive way for automated model selection. 

### Installation (work in progress)

<i>to be implemented</i>

### Usage (work in progress)
<h4>Classification</h4>
<ul>
    <li>Binary Classification</li>
   
    from exerevnetes import BinaryClassifierComparator # import

    cmp = BinaryClassifierComparator(X, y) # basic comparator initialization

    cmp.run() # run the comparator, which at the end displays the results

    cmp.get_results() # returns the results dataframe sorted by precision 

    cmp.get_best_clf(metric="precision_score") # returns the classifier that achieved the higher score on the metric (by default f1_score is being used)
</ul>
<ul>
    <li>Multiclass classification (to be implemented)</li>


## Contributions

This is an open source project and I welcome any contributions - ideas

### Local Setup - Development

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