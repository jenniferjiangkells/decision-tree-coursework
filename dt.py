import numpy as np
import matplotlib as plt

##################################################
#Decision Tree
##################################################

#Loading Data - should be passed as dictionary into decision tree?
data=np.loadtxt('wifi_db/clean_dataset.txt')
print(data)

noisydata=np.loadtxt('wifi_db/noisy_dataset.txt')
print(noisydata)

#Decision tree algorithm
def decision_tree_learning(data,depth):
    if len(np.unique(data[:, 7])) == 1:
        terminal_node = {'attribute': 'leaf', 'value': data[0, 7], 'left': {}, 'right': {}}
        return terminal_node, depth
    else:
        find_split(data)

def find_split(data):
    entrophy = get_entrophy(data)

def get_entrophy(data):
    return entrophy

#def split_data(data, index):

#Bonus question for printing tree
#def print_tree():

##################################################
#Evaluation
##################################################

classes = [1,2,3,4]

#def evaluate(test_db, trained_tree):

#def get_confusion_matrix(test, trained, classes):

#def get_precision_rate(confusion_matrix):

#def get_recall_rate(confusion_matrix):


#def get_F1:

#def get_avg_classification_rate:

##################################################
#Pruning
##################################################
