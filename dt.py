import numpy as np
import pandas as pd
import matplotlib as plt

##################################################
#Decision Tree
##################################################

#Loading Data
data=np.loadtxt('wifi_db/clean_dataset.txt')
#print(data)

noisydata=np.loadtxt('wifi_db/noisy_dataset.txt')
#print(noisydata)


def split_data(index,feature, data):

    #return dataset[:index,:], dataset[index:,:]

    left = np.where(data[:,feature] <= index)
    right = np.where(data[:,feature] > index)
    left_dataset = data[left]
    right_dataset = data[right]
    return left_dataset, right_dataset


def find_split(dataset):
    best_feature = 0
    best_gain = 0
    split_value =0
    #get total number of features in dataset
    features = dataset.shape[0] -1
    print(features)
    for feature in range(7):
        #get all unique classes for a feature
        unique_set = np.unique(dataset[:,feature])
        print(unique_set)
        print(unique_set.shape)
        for index in unique_set:
            left_dataset, right_dataset = split_data(split_value, feature,dataset)
            if len(left_dataset) == 0 or len(right_dataset) == 0:
                continue
             #calculate the information gain
            gain = get_info_gain(dataset,left_dataset,right_dataset)
            if gain > best_gain:
                best_feature = feature
                best_gain = gain
    return best_feature, best_gain

def get_entropy(dataset):
    entropy = 0
    elements,count = np.unique(dataset[:,7],return_counts = True)
    total = dataset.shape[0]
    for label in range(len(elements)):
        p = count[label]/total
        entropy -= p * np.log2(p)
    return entropy


def get_info_gain(current, left, right):
    #get the full entropy of the dataset
    total_entropy = get_entropy(current)
    total = len(current)
    gain = total_entropy - ((len(left)/total)* get_entropy(left) + (len(right)/total) * get_entropy(right))
    return gain






#Decision tree algorithm
def decision_tree_learning(dataset,depth):
    if len(np.unique(dataset[:, 7])) == 1:
        terminal_node = {'attribute': 'leaf', 'value': data[0, 7], 'left': {}, 'right': {}}
        return terminal_node, depth
    else:
        attribute, value = find_split(dataset)
        l = dataset[:, attribute] <= value
        r = dataset[:, attribute] > value
        l_dataset = dataset[l]
        r_dataset = dataset[r]
        l_branch, l_depth = decision_tree_learning(l_dataset,depth+1)
        r_branch, r_depth = decision_tree_learning(r_dataset,depth+1)
        node = {'attribute': attribute, 'value': value, 'left': l_branch, 'right': r_branch}
        return node, max(l_depth,r_depth)

dt, depth = decision_tree_learning(data,0)
print(dt)
print(depth)

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
