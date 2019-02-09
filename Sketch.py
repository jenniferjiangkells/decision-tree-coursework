for i in range(10):
    final_raw_model_array = []
    validation_raw_model_array = []
    pruned_model_array = []

    start = int(len(dataset) * i / 10)
    end = int(len(dataset) * (i + 1) / 10)
    test_data = dataset[start:end]

    training_data = dataset[:start]
    training_data.extend(dataset[end:])

    # call cv here
    cross_validation(training_data, test_data):
        final_raw_model, fina_raw_depth = decision_tree_training(training_data)
        actual_label = int(test_data[-1])
        predicted_labels = predict(test_data, final_raw_model)

        final_raw_cmat, final_raw_precision, final_raw_recall,
        final_raw_f1, final_raw_classification = get_stats(actual_labels, predicted_labels)
        return final_raw_cmat, final_raw_precision, final_raw_recall, final_raw_f1, final_raw_classification

    final_raw_cmat_sum += final_raw_cmat
    final_raw_precision_sum += final_raw_precision
    final_raw_recall_sum += final_raw_recall
    final_raw_f1_sum += final_raw_f1
    final_raw_class_rate_sum += final_raw_classification

    # call inner cv here
    Inner_validation(training_data):

        prunned_model_array = []

        for i in range(10):
            start = int(len(dataset) * i / 10)
            end = int(len(dataset) * (i + 1) / 10)
            validation_data = dataset[start:end]

            inner_training_data = dataset[:start]
            inner_training_data.extend(dataset[end:])

            # raw model before pruning
            pre_pruned_model, pre_pruned_depth = decision_tree_training(inner_training_data)

            actual_label = int(validation_data[-1])
            predicted = predict(validation_data, pre_pruned_model)

            pre_pruned_cmat, pre_pruned_precision, pre_pruned_recall,
            pre_pruned_f1, pre_pruned_classification = get_stats(actual_labels, predicted_labels)

            pre_pruned_cmat_sum += cmat
            pre_pruned_precision_sum += precision
            pre_pruned_recall_sum += recall
            pre_pruned_f1_sum += f1
            pre_pruned_class_rate_sum += classification

            # prune the model here
            pruned_mdeol, pruned_depth = prune(pre_pruned_model, validation_data, pre_pruned_model)

            prunned_model_array.append(prunned_model)

            predicted = predict(validation_data, pruned_model)

            pruned_cmat, pruned_precision, pruned_recall,
            pruned_f1, pruned_classification = get_stats(actual_labels, predicted_labels)

            pruned_cmat_sum += cmat
            pruned_precision_sum += precision
            pruned_recall_sum += recall
            pruned_f1_sum += f1
            pruned_class_rate_sum += classification

        # print out these stats inside inner cv just for comparison, we do not need to return them
        pre_pruned_avg_cm = cmat_sum / 10
        pre_pruned_avg_precision = precision_sum / 10
        pre_pruned_avg_recall = recall_sum / 10
        pre_pruned_avg_f1 = f1_sum / 10
        pre_pruned_avg_classification = class_rate_sum / 10

        pruned_avg_cm = cmat_sum / 10
        pruned_avg_precision = precision_sum / 10
        pruned_avg_recall = recall_sum / 10
        pruned_avg_f1 = f1_sum / 10
        pruned_avg_classification = class_rate_sum / 10

        return pruned_model_array

    for i in range(len(pruned_model_array))
        predicted = predict(test_data, pruned_model[i])
        # same process here for getting the stats

# This is outside the outest for loop
# gets the average stats for the 10 final raw models
final_raw_avg_cm = cmat_sum / 10
final_raw_avg_precision = precision_sum / 10
final_raw_avg_recall = recall_sum / 10
final_raw_avg_f1 = f1_sum / 10
final_raw_avg_classification = class_rate_sum / 10

# get the average stats for prunned models, same process as above