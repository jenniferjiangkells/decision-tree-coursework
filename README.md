# Decision Tree CBC

Final_Decision_Tree is preloaded with two sets of datasets (clean and noisy data on WIFI strengths in different rooms)
and will train, plot, predict, evaluate, and prune decision trees based on these datasets using 10-fold evaluation.

## Prerequisites

Our code depends on numpy and matplotlib and requires the installation of these modules,
which can be installed with the commands given below (per the specification).

```bash

export PYTHONUSERBASE=/vol/bitbucket/nuric/pypi

```

## Running the code

Our code runs on Python 3. To run, simply type

```bash

python3 Final_Decision_Tree.py

```

This will run the decision tree with the two pre-loaded datasets, clean_data.txt and noisy_data.txt from the folder wifi_db. The code will first evaluate the clean data and output information on the depth and average classification rates of the unpruned and pruned trees in each individual fold during 10-fold cross validation. It will then output the final average confusion matrix and precision, recall, F1 measure, and classification rates of the unpruned and pruned tree after evaluation. The noisy data is evaluated subsequently and outputs the same information. The code will also output a visualization of the decision tree trained on clean_data.
