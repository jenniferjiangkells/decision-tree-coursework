import numpy as np
import math
import random
import json
import matplotlib.pyplot as plt
import time
import copy

##################################################
#Decision Tree
##################################################

#Loading the data
clean_data = np.loadtxt("./wifi_db/clean_dataset.txt")
noisy_data = np.loadtxt("./wifi_db/noisy_dataset.txt")

# Func to calculate entropy of dataset
def get_entropy(dataset):
    total = len(dataset)
    count_dict = {}

    for data in dataset:
        label = int(data[-1])
        count_dict[label] = count_dict.get(label, 0) + 1

    entropy = 0
    # print(count_dict)
    for label, count in count_dict.items():
        p = count / total
        entropy = entropy + p * math.log(p, 2)
    return -entropy


# Func to calculate entropy after splitting
def remainder_entropy(ldata, rdata):
    left_entropy = get_entropy(ldata)
    right_entropy = get_entropy(rdata)

    left_size = len(ldata)
    right_size = len(rdata)

    return (left_size * left_entropy + right_size * right_entropy) / (left_size + right_size)

# Find best split point
def find_split(training_data):
    if len(training_data) == 0:
        return 0, 0, [], []
    attributes_length = len(training_data[0]) - 1
    best_information_gain = 0
    best_split = (0, 0, [], [])

    base_entropy = get_entropy(training_data)

    for attribute_index in range(attributes_length):
        # sort the data based on each attribute
        sorted_data = sorted(training_data, key=lambda x : x[attribute_index])

        # find the best value of the attribute that splits label
        value_set = set()
        values = []
        for value in sorted_data:
            values.append(value[attribute_index])
        #values = [value[attribute_index]for value in sorted_data]

        for value in values:
            value = int(value)
            if value in value_set:
                continue
            value_set.add(value)

            # Assume split the data by the value, calculate the information gain
            # Split data by data > value as right, otherwise left
            ldata = []
            rdata = []
            for data in sorted_data:
                if data[attribute_index] > value:
                    rdata.append(data)
                else:
                    ldata.append(data)


            remainder = remainder_entropy(ldata, rdata)
            information_gain = base_entropy - remainder

            # Replace with better information gain
            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_split = (attribute_index, value, ldata, rdata)

    return best_split

np.random.seed(0)

# Train the decision tree model
def decision_tree_training(training_data, depth=0):
    if len(training_data) == 0:
        return None, 0

    first_label = int(training_data[0][-1])
    all_same = True

    for data in training_data:
        label = int(data[-1])
        if label != first_label:
            all_same = False

    if all_same:
        # leaf node no care for attribute and value, since they all have same label
        return {'attribute': 0, 'value': 0,'left': None, 'right': None, 'length': len(training_data), 'label': first_label}, depth

    # Find out the best split point
    attribute, value, ldata, rdata = find_split(training_data)
    root_node = {'attribute': attribute, 'value': value, 'length': len(training_data)}

    # Recursively build left and right tree
    lnode, ldepth = decision_tree_training(ldata, depth + 1)
    rnode, rdepth = decision_tree_training(rdata, depth + 1)
    root_node['left'] = lnode
    root_node['right'] = rnode

    return root_node, max(ldepth, rdepth)

def predict(data, model):
    tree = model
    while True:
        if tree['left'] is None and tree['right'] is None:
            return tree['label']
        attribute = tree['attribute']
        value = tree['value']

        if data[attribute] > value:
            tree = tree['right']
        else:
            tree = tree['left']

    return -1

# Evaluate accuracy of model
def test_over_model(tree_model, test_data):
    actual_labels=[]
    predicted_labels = []

    for data in test_data:
        label = int(data[-1])
        predicted = predict(data, tree_model)
        actual_labels.append(label)
        predicted_labels.append(predicted)

    cmat = get_confusion_matrix(actual_labels,predicted_labels)
    class_rate = classification_rate(cmat)

    return class_rate

##################################################
#Evaluation
##################################################

# Evaluate accuracy, precision, recall and F1 score of dataset
def evaluate(dataset):
    #shuffle the data first
    random.shuffle(dataset)

    models_array = []

    cmat_sum = np.zeros((4,4))
    precision_sum = np.zeros((4))
    recall_sum = np.zeros((4))
    f1_sum = np.zeros((4))
    class_rate_sum = 0

    for i in range(10):
        actual_labels=[]
        predicted_labels = []

        start = int(len(dataset) * i / 10)
        end = int(len(dataset) * (i + 1) / 10)
        validation_data = dataset[start:end]

        training_data = dataset[:start]
        training_data.extend(dataset[end:])

        model, depth = decision_tree_training(training_data)
        models_array.append(model)

        for data in validation_data:
            label = int(data[-1])
            predicted = predict(data, model)
            actual_labels.append(label)
            predicted_labels.append(predicted)

        cmat = get_confusion_matrix(actual_labels,predicted_labels)
        precision = get_precision(cmat)
        recall = get_recall(cmat)
        cmat_sum += cmat
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1_measure(precision, recall)
        class_rate_sum += classification_rate(cmat)

    print("Average confusion matrix:\n", cmat_sum/10)
    print("Average precision rate: \n", precision_sum/10)
    print("Average recall rate: \n", recall_sum/10)
    print("Average F1 measure: \n", f1_sum/10)
    print("Average classification rate: \n", class_rate_sum/10)

    return models_array

def get_stats(actual_labels, predicted_labels):
    cmat_sum = np.zeros((4,4))
    precision_sum = np.zeros((4))
    recall_sum = np.zeros((4))
    f1_sum = np.zeros((4))
    class_rate_sum = 0

    cmat = get_confusion_matrix(actual_labels,predicted_labels)
    precision = get_precision(cmat)
    recall = get_recall(cmat)
    f1_rate = f1_measure(precision, recall)
    class_rate = classification_rate(cmat)

    return cmat, precision, recall, f1_rate, class_rate


def get_confusion_matrix(actual_labels, predicted_labels):
    cmat = np.zeros((4,4))
    for i in range(len(predicted_labels)):
        cmat[actual_labels[i] -1, predicted_labels[i] -1] += 1
    return cmat

def get_recall(confusion_matrix):
    rate = np.zeros((4))
    # compute the recall rate for each class
    for i in range(4):
        if sum(confusion_matrix[i, :]) == 0:
            rate[i] = confusion_matrix[i, i] * 100
        else:
            rate[i] = confusion_matrix[i, i] / (sum(confusion_matrix[i, :])) *100
    return rate

def get_precision(confusion_matrix):
    rate = np.zeros((4))
    for i in range(4):
        if sum(confusion_matrix[:, i]) == 0:
            rate[i] = confusion_matrix[i, i] * 100
        else:
            rate[i] = confusion_matrix[i, i] * 100 / (sum(confusion_matrix[:, i]))
    return rate

def f1_measure(precision_rate, recall_rate):
    rate = np.zeros((4))
    for i in range(4):
        if precision_rate[i] == 0 or recall_rate[i] == 0:
            rate[i] = 0
        else:
            rate[i] = 2 * ((precision_rate[i]/100 * recall_rate[i]/100) / (precision_rate[i]/100 + recall_rate[i]/100))*100
    return rate

def classification_rate(confusion_matrix):
    rate = sum(confusion_matrix.diagonal()) / confusion_matrix.sum()
    return rate


##################################################
#Plotting (Bonus)
##################################################

#plot the tree
decision_node = dict(boxstyle="square",fc="w")
leaf_node = dict(boxstyle="square",fc="w")
arrow_args = dict(arrowstyle="<-")

def get_tree_width(tree):
	leaf_num = 0
	if tree['left'] is None:
		return 0, 0
	leftL, leftR = get_tree_width(tree['left'])
	rightL, rightR = get_tree_width(tree['right'])

	left = min(leftL - 1, rightL + 1)
	right = max(leftR - 1, rightR + 1)

	return left, right

def get_tree_depth(tree, depth=1):
    if tree is None:
        return depth
    ldepth = get_tree_depth(tree['left'], depth + 1)
    rdepth = get_tree_depth(tree['right'], depth + 1)

    return max(ldepth, rdepth)

def plotNode(nodeTxt, centerPt, parentPt, nodeType, ax1):
	plt.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
							xytext=centerPt, textcoords='axes fraction', fontsize=60,
							va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)
def plotMidText(cntrPt, parentPt, txtString, ax1):
	xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
	yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
	plt.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, depth, parentPt, nodeTxt, ax1, params):
	left, right = get_tree_width(myTree)
	numLeafs = right - left

	cntrPt = (params['xOff'] + (1.0 + float(numLeafs)) / params['totalW'] * 8, params['yOff'])
	plotMidText(cntrPt, parentPt, nodeTxt, ax1)
	plotNode(nodeTxt, cntrPt, parentPt, decision_node, ax1)

	params['yOff'] = params['yOff'] - 1.0

	if not myTree['left'] is None:
		plotTree(myTree['left'], depth - 1, cntrPt, '{} <= {}'.format(myTree['attribute'], myTree['value']), ax1, params)
		plotTree(myTree['right'], depth - 1, cntrPt, '{} > {}'.format(myTree['attribute'], myTree['value']), ax1, params)
	else:
		params['xOff'] = params['xOff'] + 8.0 / params['totalW']
		plotNode('label = {}'.format(myTree['label']), (params['xOff'], params['yOff']), cntrPt, leaf_node, ax1)
		plotMidText((params['xOff'], params['yOff']), cntrPt, 'label = {}'.format(myTree['label']), ax1)
	params['yOff'] = params['yOff'] + 1.0

def createPlot(myTree):
	fig = plt.figure(1, facecolor='white')
	fig.clf()
	depth = 10
	axprops = dict(xticks=[], yticks=[])
	ax1 = plt.subplot(111, frameon=False, **axprops)
	left, right = get_tree_width(myTree)

	params = {}
	params['totalW'] = float(right - left)
	params['totalD'] = float(depth)
	params['xOff'] = -1
	params['yOff'] = 1.0
	plotTree(myTree, depth, (0.5, 1.0), 'root', ax1, params)

	plt.show()

##################################################
#Pruning
##################################################

# Next we do pruning on first trained tree
def prune(decision_tree, test_data, root):
	if decision_tree is None:
		return decision_tree
	accuracy = test_over_model(root, test_data)

	if decision_tree['left'] is None or decision_tree['right'] is None:
		return

	# Can be pruned
	if 'label' in decision_tree['left'] or 'label' in decision_tree['right']:
		count1 = decision_tree['left'].get('length', 0)
		label1 = decision_tree['left'].get('label', 0)

		count2 = decision_tree['right'].get('length', 0)
		label2 = decision_tree['right'].get('label', 0)

		label = label1
		count = count1
		if count2 > count1:
			label = label2
			count = count2
		decision_tree['left'] = None
		decision_tree['right'] = None
		decision_tree['attribute'] = 0
		decision_tree['value'] = 0
		decision_tree['label'] = label
		decision_tree['length'] = count
		return decision_tree

	left_tmp = copy.deepcopy(decision_tree['left'])
	right_tmp = copy.deepcopy(decision_tree['right'])

	left_pruned = prune(decision_tree['left'], test_data, root)
	right_pruned = prune(decision_tree['right'], test_data, root)

	decision_tree['left'] = left_pruned
	decision_tree['right'] = right_pruned

	new_accuracy = test_over_model(root, test_data)

	# If got better result, apply the changes
	if new_accuracy >= accuracy:
		print('New accuracy %f old accuracy: %f' % (new_accuracy, accuracy))
		return decision_tree
	else:
		print('Worse accuracy %f old accuracy %f' % (new_accuracy, accuracy))
		decision_tree['left'] = left_tmp
		decision_tree['right'] = right_tmp
		return decision_tree


# Evaluate on cleaned data
models = evaluate(clean_data.tolist())

# Plot the diagram
# createPlot(models[0])

# Prune on first model
start_index = int(len(clean_data) * 0.1)
test_data = clean_data[:start_index]

print('Max depth of original tree= %d' % get_tree_depth(models[0]))
print('Start pruning on first model')
pruned_tree = prune(models[0], test_data, models[0])
class_rate = test_over_model(pruned_tree, test_data)
print('New accuracy after prune %f' % class_rate)
print('Max depth of pruned tree= %d' % get_tree_depth(pruned_tree))

print('\n\n')


# Evaluate on noisy data
print('Evaluate on noisy dataset')
models = evaluate(noisy_data.tolist())

# Plot the tree diagram
#createPlot(models[0])

# Prune on first model
start_index = int(len(noisy_data) * 0.1)
test_data = noisy_data[:start_index]

print('Max depth of original tree= %d' % get_tree_depth(models[0]))
print('Start pruning on first model')
pruned_tree = prune(models[0], test_data, models[0])
class_rate = test_over_model(pruned_tree, test_data)
print('New accuracy after prune %f' % class_rate)
print('Max depth of pruned tree= %d' % get_tree_depth(pruned_tree))

print('$$$$$')
