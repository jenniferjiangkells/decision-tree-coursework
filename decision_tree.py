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
    left = np.where(data[:,feature] <= index)
    right = np.where(data[:,feature] > index)
    left_dataset = data[left]
    right_dataset = data[right]
    return left_dataset, right_dataset


def find_split(dataset):
    #initialize values
    best_feature, best_value, best_gain = 0, 0.0, 0.0
    for feature in range(7):
        #get all unique classes for a feature
        unique_set = np.unique(dataset[:,feature])
        for index in unique_set:
            left_dataset, right_dataset = split_data(index,feature,dataset)
            #calculate the information gain
            gain = get_info_gain(dataset, left_dataset, right_dataset)
            if len(left_dataset)==0 or len(right_dataset)==0:
               continue
            if gain > best_gain:
                best_feature = feature +1
                best_value = index
                best_gain = gain
    return best_feature, best_value

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
    total = current.shape[0]
    gain = total_entropy - ((left.shape[0]/total)* get_entropy(left) + (right.shape[0]/total) * get_entropy(right))
    return gain

np.random.seed(0)

#Decision tree algorithm
def decision_tree_learning(dataset,depth):
    if len(np.unique(dataset[:, 7])) == 1:
        attribute = np.unique(dataset[:,7])
        terminal_node = {'attribute':int(attribute),'leaf':1,'value':0,'left': None,'right': None}
        return terminal_node, depth
    else:
        attribute, value = find_split(dataset)
        l =np.where(dataset[:, attribute-1] <= value)
        r =np.where(dataset[:, attribute-1] > value)
        l_dataset = dataset[l]
        r_dataset = dataset[r]
        l_branch, l_depth = decision_tree_learning(l_dataset,depth+1)
        r_branch, r_depth = decision_tree_learning(r_dataset,depth+1)
        node = {'attribute': attribute,'leaf': 0, 'value': value, 'left': l_branch, 'right': r_branch}
        return node, max(l_depth,r_depth)


dt, depth = decision_tree_learning(data,0)
print(depth)
print(dt)



#Bonus question for printing tree

def visualize(dt):
    for key in dt.keys():
        if type(dt[key]).__name__ == 'dict':
            visualize(dt[key])
        else:
            print(key, ":", dt[key])

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
