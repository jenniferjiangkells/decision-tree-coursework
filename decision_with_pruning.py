import numpy as np
import math
import random
import json
import matplotlib.pyplot as plt
import time
import copy

#Import packages for plotting tree graph
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from collections import deque

##################################################
#Decision Tree
##################################################

#Loading the data
clean_data = np.loadtxt("./wifi_db/clean_dataset.txt")
noisy_data = np.loadtxt("./wifi_db/noisy_dataset.txt")


# Func to calculate entropy of dataset
def get_entropy(dataset):
	total = len(dataset)
	label_dict = {}   #labels and counts will be stored in this dictionary

	for data in dataset:
		label = int(data[-1])
		label_dict[label] = label_dict.get(label, 0) + 1

	entropy = 0

	for label, count in label_dict.items():
		p = count / total
		entropy = entropy + p * math.log(p, 2)
	return -entropy


# This function calculates the entropy after splitting
def remainder_entropy(ldata, rdata):
	left_entropy = get_entropy(ldata)
	right_entropy = get_entropy(rdata)

	left_size = len(ldata)
	right_size = len(rdata)

	return (left_size * left_entropy + right_size * right_entropy) / (left_size + right_size)


# Find best split point
def find_split(training_data):
	if len(training_data) == 0:
		return 0, 0, [], []       #returns a tuple of attribute, splitting value, left_data and right_data
	attributes_length = len(training_data[0]) - 1   #returns number of attributes in the data
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

		for value in values:
			value = int(value)
			if value in value_set:
				continue
			value_set.add(value)

			# Assume split the data by the value, calculate the information gain
			# Split data > value as right, otherwise left
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
def decision_tree_training(training_data, depth=1):
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


# Evaluate accuracy(classification rate) of the input model.
def evaluate(tree_model, test_data):
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


#this function generates ten pruned models.
def Inner_validation(dataset):
	pruned_models_array = []
	pre_pruned_depth_array = []
	pruned_depth_array = []
	pruned_models = []

	pre_pruned_class_rate_array = []
	pruned_class_rate_array = []

	for i in range(10):
		pre_pruned_actual_labels=[]
		pre_pruned_predicted_labels = []

		pruned_actual_labels = []
		pruned_predicted_labels = []

		start = int(len(dataset) * i / 10)
		end = int(len(dataset) * (i + 1) / 10)
		validation_data = dataset[start:end]

		training_data = dataset[:start]
		training_data.extend(dataset[end:])

		# get the pre_pruned_model and their stats
		pre_pruned_model, pre_pruned_depth = decision_tree_training(training_data, depth = 1)

		pre_pruned_class_rate = evaluate(pre_pruned_model, validation_data)

		pre_pruned_class_rate_array.append(pre_pruned_class_rate)

		pre_pruned_depth_array.append(pre_pruned_depth)

		# get the corresponding pruned model and their stats
		pruned_model = prune(pre_pruned_model, validation_data, pre_pruned_model)
		pruned_depth = get_tree_depth(pruned_model, depth = 1)

		pruned_depth_array.append(pruned_depth)

		pruned_class_rate = evaluate(pruned_model, validation_data)
		pruned_class_rate_array.append(pruned_class_rate)
		pruned_models.append(pruned_model)

	print("average pre_pruned tree depth is: \n", pre_pruned_depth_array)
	print("average pruned tree depth is: \n", pruned_depth_array)
	print("average pre_pruned class rate for 10 models in inner cv: \n", np.average(pre_pruned_class_rate_array))
	print("average pruned class rate for 10 models in inner cv: \n", np.average(pruned_class_rate_array))
	print("\n")
	# print("Average depth of the ten pruned trees are: ", np.average(pruned_depth_array))

	return pruned_models


# Evaluate accuracy, precision, recall and F1 score of dataset
# This cross validation function takes in dataset, and essentially divide the dataset into ten folds to perform the
# cross validation. If Pruned_or_Raw is 0, this function will return an array of ten raw models with their average stats at the end.
# These ten models are basically produced
# from ten different training dataset. If Pruned_or_Raw is 1, this means we are trying to do cv on pruned models. For every training
# dataset(there will be 10 different training datasets), we will run another 10 fold cv on it by passing the training dataset into
# the inner_validation function, which returns ten pruned models.

def cross_validation(training_data, test_data):
	models_array = []
	depth_array = []

	predicted_labels = []
	actual_labels = []

	final_raw_model, final_raw_depth = decision_tree_training(training_data)

	for data in test_data:
		actual_label = int(data[-1])
		predicted = predict(data, final_raw_model)
		actual_labels.append(actual_label)
		predicted_labels.append(predicted)

	final_raw_cmat, final_raw_precision, final_raw_recall, final_raw_f1, \
	final_raw_classification = get_stats(actual_labels, predicted_labels)

	return final_raw_cmat, final_raw_precision, final_raw_recall, final_raw_f1, final_raw_classification, final_raw_depth


#helper functions to calculate stats and confusion matrix
#########################################################
def get_stats(actual_labels, predicted_labels):

	cmat = get_confusion_matrix(actual_labels,predicted_labels)
	precision = get_precision(cmat)
	recall = get_recall(cmat)
	f1_rate = f1_measure(precision, recall)
	class_rate = classification_rate(cmat)

	return cmat, precision, recall, f1_rate, class_rate


def get_confusion_matrix(actual_labels, predicted_labels):
	#initialize confusion matrix
	cmat = np.zeros((4,4))
	#loops through data and counts number of actual-prediction occurences to create confusion matrix
	for i in range(len(predicted_labels)):
		cmat[actual_labels[i] -1, predicted_labels[i] -1] += 1

	return cmat


def get_recall(confusion_matrix):
	recall = np.zeros((4))

	# recall is TP of given class over TP+FP
	for i in range(4):
		if sum(confusion_matrix[i, :]) == 0:
			recall[i] = confusion_matrix[i, i]
		else:
			recall[i] = confusion_matrix[i, i] / (sum(confusion_matrix[i, :]))

	return recall * 100

def get_precision(confusion_matrix):
	precision = np.zeros((4))

	for i in range(4):
		if sum(confusion_matrix[:, i]) == 0:
			precision[i] = confusion_matrix[i, i]
		else:
			precision[i] = confusion_matrix[i, i] / (sum(confusion_matrix[:, i]))

	return precision * 100


def f1_measure(precision_rate, recall_rate):
	f1 = np.zeros((4))
	#calculate the f1 measure for every class using precision and recall rates
	for i in range(4):
		f1[i] = 2 * ((precision_rate[i] * recall_rate[i]) / (precision_rate[i] + recall_rate[i]))

	return f1


def classification_rate(confusion_matrix):
	#the classification rate is the diagonal of the confusion matrix (TP+TN) over total
	classification = sum(confusion_matrix.diagonal()) / confusion_matrix.sum()

	return classification


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
		depth = depth - 1
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

##########################################
#Use this plot function if the above doesn't run on Linux
###########################################

#fig, ax = plt.subplots(figsize=(18, 10))

#tree, depth = decision_tree_training(clean_data)

#gap = 1.0/depth

#def plot_graph(root, xmin, xmax, ymin, ymax):
#  queue = deque([(root, xmin, xmax, ymin, ymax)])
#  while len(queue) > 0:
#    q = queue.popleft()
#    node = q[0]
#    xmin = q[1]
#    xmax = q[2]
#    ymin = q[3]
#    ymax = q[4]
#    atri = node['attribute']
#    val = node['value']
#    text = '['+str(atri)+']:'+str(val)

#    center = xmin+(xmax-xmin)/2.0
#    d = (center-xmin)/2.0

#    if node['left'] != None:
#      queue.append((node['left'], xmin, center, ymin, ymax-gap))
#      ax.annotate(text, xy=(center-d, ymax-gap), xytext=(center, ymax),arrowprops=dict(arrowstyle="->"),)

#    if node['right'] != None:
#      queue.append((node['right'], center, xmax, ymin, ymax-gap))
#      ax.annotate(text, xy=(center+d, ymax-gap), xytext=(center, ymax),arrowprops=dict(arrowstyle="->"),)

#    if node['left'] is None and node['right'] is None:
#      an1 = ax.annotate(node['label'], xy=(center, ymax), xycoords="data", va="bottom", ha="center",
#                        bbox=dict(boxstyle="round", fc="w"))

#plot_graph(tree, 0.0, 1.0, 0.0, 1.0)

#fig.subplots_adjust(top=0.83)
#plt.show()

##################################################
#Pruning
##################################################

# Next we do pruning on trained tree
def prune(decision_tree, test_data, root):      #the inner cross validation is just the dataset we pass into prune function
	if decision_tree is None:
		return decision_tree
	accuracy = evaluate(root, test_data)

	if decision_tree['left'] is None and decision_tree['right'] is None:
		return decision_tree

	# Can be pruned
	if 'label' in decision_tree['left'] and 'label' in decision_tree['right']:
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

	new_accuracy = evaluate(root, test_data)

	# If got better result, apply the changes
	if new_accuracy >= accuracy:
		# print('New accuracy %f old accuracy: %f' % (new_accuracy, accuracy))
		return decision_tree
	else:
		# print('Worse accuracy %f old accuracy %f' % (new_accuracy, accuracy))
		decision_tree['left'] = left_tmp
		decision_tree['right'] = right_tmp
		return decision_tree




# Evaluate on cleaned data
print("Evaluate on cleaned data")

np.random.shuffle(clean_data)

final_raw_depth_array = []
pruned_depth_array = []

final_raw_cmat_sum = np.zeros((4, 4))
final_raw_precision_sum = np.zeros((4))
final_raw_recall_sum = np.zeros((4))
final_raw_f1_sum = np.zeros((4))
final_raw_class_rate_sum = 0

pruned_cmat_sum = np.zeros((4, 4))
pruned_precision_sum = np.zeros((4))
pruned_recall_sum = np.zeros((4))
pruned_f1_sum = np.zeros((4))
pruned_class_rate_sum = 0

for i in range(10):

	start = int(len(clean_data) * i / 10)
	end = int(len(clean_data) * (i + 1) / 10)
	test_data = clean_data[start:end]

	training_data1 = clean_data[:start]
	training_data2 = clean_data[end:]
	training_data = np.concatenate([training_data1, training_data2])

	final_raw_cmat, final_raw_precision, final_raw_recall, final_raw_f1, \
	final_raw_classification, final_raw_depth = cross_validation(training_data, test_data)

	final_raw_depth_array.append(final_raw_depth)

	final_raw_cmat_sum += final_raw_cmat
	final_raw_precision_sum += final_raw_precision
	final_raw_recall_sum += final_raw_recall
	final_raw_f1_sum += final_raw_f1
	final_raw_class_rate_sum += final_raw_classification

	print("In fold ", i+1)

	pruned_models = Inner_validation(training_data.tolist())

	#createPlot(pruned_models[0])

	for i in range(len(pruned_models)):
		pruned_actual_labels = []
		pruned_predicted_labels = []

		for data in test_data:
			label = int(data[-1])
			predicted = predict(data, pruned_models[i])
			pruned_actual_labels.append(label)
			pruned_predicted_labels.append(predicted)

		pruned_cmat, pruned_precision, pruned_recall, pruned_f1, \
		pruned_classification = get_stats(pruned_actual_labels, pruned_predicted_labels)

		pruned_cmat_sum += pruned_cmat
		pruned_precision_sum += pruned_precision
		pruned_recall_sum += pruned_recall
		pruned_f1_sum += pruned_f1
		pruned_class_rate_sum += pruned_classification

print("Average final raw confusion matrix:\n", final_raw_cmat_sum / 10)
print("Average final raw precision rate: \n", final_raw_precision_sum / 10)
print("Average final raw recall rate: \n", final_raw_recall_sum / 10)
print("Average final raw F1 measure: \n", final_raw_f1_sum / 10)
print("Average final raw classification rate: \n", final_raw_class_rate_sum / 10)
print("Average depth of the ten final raw trees are: ", np.average(final_raw_depth_array))
print("\n")
print("Average pruned confusion matrix:\n", pruned_cmat_sum / 100)
print("Average pruned precision rate: \n", pruned_precision_sum / 100)
print("Average pruned recall rate: \n", pruned_recall_sum / 100)
print("Average pruned F1 measure: \n", pruned_f1_sum / 100)
print("Average pruned classification rate: \n", pruned_class_rate_sum / 100)
# print("Average depth of the ten pruned trees are: ", np.average(final_raw_depth_array))

print('\n\n')


#Evaluate on noisy data
print('Evaluate on noisy dataset')

np.random.shuffle(noisy_data)

final_raw_depth_array = []
pruned_depth_array = []

final_raw_cmat_sum = np.zeros((4, 4))
final_raw_precision_sum = np.zeros((4))
final_raw_recall_sum = np.zeros((4))
final_raw_f1_sum = np.zeros((4))
final_raw_class_rate_sum = 0

pruned_cmat_sum = np.zeros((4, 4))
pruned_precision_sum = np.zeros((4))
pruned_recall_sum = np.zeros((4))
pruned_f1_sum = np.zeros((4))
pruned_class_rate_sum = 0

for i in range(10):
	start = int(len(noisy_data) * i / 10)
	end = int(len(noisy_data) * (i + 1) / 10)
	test_data = clean_data[start:end]

	training_data1 = noisy_data[:start]
	training_data2 = noisy_data[end:]
	training_data = np.concatenate([training_data1, training_data2])

	final_raw_cmat, final_raw_precision, final_raw_recall, final_raw_f1, \
	final_raw_classification, final_raw_depth = cross_validation(training_data, test_data)

	final_raw_depth_array.append(final_raw_depth)

	final_raw_cmat_sum += final_raw_cmat
	final_raw_precision_sum += final_raw_precision
	final_raw_recall_sum += final_raw_recall
	final_raw_f1_sum += final_raw_f1
	final_raw_class_rate_sum += final_raw_classification

	print("In fold ", i+1)

	pruned_models = Inner_validation(training_data.tolist())

	for i in range(len(pruned_models)):
		pruned_actual_labels = []
		pruned_predicted_labels = []

		for data in test_data:
			label = int(data[-1])
			predicted = predict(data, pruned_models[i])
			pruned_actual_labels.append(label)
			pruned_predicted_labels.append(predicted)

		pruned_cmat, pruned_precision, pruned_recall, pruned_f1, \
		pruned_classification = get_stats(pruned_actual_labels, pruned_predicted_labels)

		pruned_cmat_sum += pruned_cmat
		pruned_precision_sum += pruned_precision
		pruned_recall_sum += pruned_recall
		pruned_f1_sum += pruned_f1
		pruned_class_rate_sum += pruned_classification

print("Average final raw confusion matrix:\n", final_raw_cmat_sum / 10)
print("Average final raw precision rate: \n", final_raw_precision_sum / 10)
print("Average final raw recall rate: \n", final_raw_recall_sum / 10)
print("Average final raw F1 measure: \n", final_raw_f1_sum / 10)
print("Average final raw classification rate: \n", final_raw_class_rate_sum / 10)
print("Average depth of the ten final raw trees are: ", np.average(final_raw_depth_array))
print("\n")
print("Average pruned confusion matrix:\n", pruned_cmat_sum / 100)
print("Average pruned precision rate: \n", pruned_precision_sum / 100)
print("Average pruned recall rate: \n", pruned_recall_sum / 100)
print("Average pruned F1 measure: \n", pruned_f1_sum / 100)
print("Average pruned classification rate: \n", pruned_class_rate_sum / 100)
# print("Average depth of the ten pruned trees are: ", np.average(final_raw_depth_array))

print('\n\n')


## first, we should use cross validation on the final test set, and generate confusion matrix and classification rate for that.
## for the remaining training set, we also do cross validation(basically training and validation set) on our models
## we just need average confusion matrix and class rate for the final test set.
