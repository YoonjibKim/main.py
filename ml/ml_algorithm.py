from constant import ml
from ml import k_means, ada_boost, db_scan, decision_tree, gaussian_mixture, gradient_boost, hierarchy, \
    k_nearest_neighbor, random_forest, logistic_regression, gaussian_naive_bayes, support_vector_machine, perceptron
import numpy as np
import copy
from sklearn.metrics import classification_report
import pandas as pd
import sys
sys.path.insert(1, './utility')
from utility import feature_management, function

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=UndefinedMetricWarning)


def get_feature_index(feature_type):
    ret_no = -1
    if feature_type is ml.feature_types[0]:
        ret_no = 0
    elif feature_type is ml.feature_types[1]:
        ret_no = 1
    elif feature_type is ml.feature_types[2]:
        ret_no = 2
    elif feature_type is ml.feature_types[3]:
        ret_no = 3
    elif feature_type is ml.feature_types[4]:
        ret_no = 4
    elif feature_type is ml.feature_types[5]:
        ret_no = 5
    elif feature_type is ml.feature_types[6]:
        ret_no = 6
    elif feature_type is ml.feature_types[7]:
        ret_no = 7
    elif feature_type is ml.feature_types[8]:
        ret_no = 8
    elif feature_type is ml.feature_types[9]:
        ret_no = 9
    elif feature_type is ml.feature_types[10]:
        ret_no = 10
    elif feature_type is ml.feature_types[11]:
        ret_no = 11
    else:
        print('no selected feature type')

    return ret_no


def ml_select(param_clustering_normal_feature_array, param_clustering_normal_label_list,
              param_clustering_attack_feature_array, param_clustering_attack_label_list,
              param_training_normal_feature_array, param_training_normal_label_list,
              param_training_attack_feature_array, param_training_attack_label_list,
              param_testing_normal_feature_array, param_testing_normal_label_list,
              param_testing_attack_feature_array, param_testing_attack_label_list,
              param_chosen_ml, param_axis_value_list, param_feature_comb_list):
    ret_ml_score_list = ['Invalid ML', -1]

    if param_chosen_ml is ml.ml_name_list[0]:
        score = k_means.run(param_clustering_normal_feature_array, param_clustering_normal_label_list,
                            param_clustering_attack_feature_array, param_clustering_attack_label_list,
                            param_axis_value_list)
        ret_ml_score_list = [param_chosen_ml, score]
    elif param_chosen_ml is ml.ml_name_list[1]:
        f1_score = ada_boost.run(param_training_normal_feature_array, param_training_normal_label_list,
                              param_training_attack_feature_array, param_training_attack_label_list,
                              param_testing_normal_feature_array, param_testing_normal_label_list,
                              param_testing_attack_feature_array, param_testing_attack_label_list,
                              param_axis_value_list)
        ret_ml_score_list = [param_chosen_ml, f1_score]
    elif param_chosen_ml is ml.ml_name_list[2]:
        score = db_scan.run(param_clustering_normal_feature_array, param_clustering_normal_label_list,
                            param_clustering_attack_feature_array, param_clustering_attack_label_list,
                            param_axis_value_list)
        ret_ml_score_list = [param_chosen_ml, score]
    elif param_chosen_ml is ml.ml_name_list[3]:
        score = decision_tree.run(param_training_normal_feature_array, param_training_normal_label_list,
                                  param_training_attack_feature_array, param_training_attack_label_list,
                                  param_testing_normal_feature_array, param_testing_normal_label_list,
                                  param_testing_attack_feature_array, param_testing_attack_label_list,
                                  param_axis_value_list)
        ret_ml_score_list = [param_chosen_ml, score]
    elif param_chosen_ml is ml.ml_name_list[4]:
        score = gaussian_mixture.run(param_clustering_normal_feature_array, param_clustering_normal_label_list,
                                     param_clustering_attack_feature_array, param_clustering_attack_label_list,
                                     param_axis_value_list)
        ret_ml_score_list = [param_chosen_ml, score]
    elif param_chosen_ml is ml.ml_name_list[5]:
        score = gradient_boost.run(param_training_normal_feature_array, param_training_normal_label_list,
                                   param_training_attack_feature_array, param_training_attack_label_list,
                                   param_testing_normal_feature_array, param_testing_normal_label_list,
                                   param_testing_attack_feature_array, param_testing_attack_label_list,
                                   param_axis_value_list)
        ret_ml_score_list = [param_chosen_ml, score]
    elif param_chosen_ml is ml.ml_name_list[6]:
        score = hierarchy.run(param_clustering_normal_feature_array, param_clustering_normal_label_list,
                              param_clustering_attack_feature_array, param_clustering_attack_label_list,
                              param_axis_value_list)
        ret_ml_score_list = [param_chosen_ml, score]
    elif param_chosen_ml is ml.ml_name_list[7]:
        score = k_nearest_neighbor.run(param_training_normal_feature_array, param_training_normal_label_list,
                                       param_training_attack_feature_array, param_training_attack_label_list,
                                       param_testing_normal_feature_array, param_testing_normal_label_list,
                                       param_testing_attack_feature_array, param_testing_attack_label_list,
                                       param_axis_value_list)
        ret_ml_score_list = [param_chosen_ml, score]
    elif param_chosen_ml is ml.ml_name_list[8]:
        score = perceptron.run(param_training_normal_feature_array, param_training_normal_label_list,
                               param_training_attack_feature_array, param_training_attack_label_list,
                               param_testing_normal_feature_array, param_testing_normal_label_list,
                               param_testing_attack_feature_array, param_testing_attack_label_list,
                               param_axis_value_list)
        ret_ml_score_list = [param_chosen_ml, score]
    elif param_chosen_ml is ml.ml_name_list[9]:
        score = random_forest.run(param_training_normal_feature_array, param_training_normal_label_list,
                                  param_training_attack_feature_array, param_training_attack_label_list,
                                  param_testing_normal_feature_array, param_testing_normal_label_list,
                                  param_testing_attack_feature_array, param_testing_attack_label_list,
                                  param_axis_value_list)
        ret_ml_score_list = [param_chosen_ml, score]
    elif param_chosen_ml is ml.ml_name_list[10]:
        score = logistic_regression.run(param_training_normal_feature_array, param_training_normal_label_list,
                                        param_training_attack_feature_array, param_training_attack_label_list,
                                        param_testing_normal_feature_array, param_testing_normal_label_list,
                                        param_testing_attack_feature_array, param_testing_attack_label_list,
                                        param_axis_value_list)
        ret_ml_score_list = [param_chosen_ml, score]
    elif param_chosen_ml is ml.ml_name_list[11]:
        score = gaussian_naive_bayes.run(param_training_normal_feature_array, param_training_normal_label_list,
                                         param_training_attack_feature_array, param_training_attack_label_list,
                                         param_testing_normal_feature_array, param_testing_normal_label_list,
                                         param_testing_attack_feature_array, param_testing_attack_label_list,
                                         param_axis_value_list)
        ret_ml_score_list = [param_chosen_ml, score]
    elif param_chosen_ml is ml.ml_name_list[12]:
        score = support_vector_machine.run(param_training_normal_feature_array, param_training_normal_label_list,
                                           param_training_attack_feature_array, param_training_attack_label_list,
                                           param_testing_normal_feature_array, param_testing_normal_label_list,
                                           param_testing_attack_feature_array, param_testing_attack_label_list,
                                           param_axis_value_list)
        ret_ml_score_list = [param_chosen_ml, score]
    else:
        print('Invalid ML')

    return ret_ml_score_list


def get_chosen_field_array(param_feature_comb_list, param_feature_array):
    column_list = []
    param_feature_array = np.array(param_feature_array)
    for feature_type in list(param_feature_comb_list):
        index = get_feature_index(feature_type)
        column_list.append(param_feature_array[:, index])

    return np.array(column_list).T


def run_score(param_chosen_ml, param_axis_value_list, param_clustering_normal_feature_array,
              param_clustering_normal_label_list, param_clustering_attack_feature_array,
              param_clustering_attack_label_list, param_training_normal_feature_array,
              param_training_normal_label_list, param_training_attack_feature_array,
              param_training_attack_label_list, param_testing_normal_feature_array,
              param_testing_normal_label_list, param_testing_attack_feature_array,
              param_testing_attack_label_list, param_feature_comb_list, param_size, param_dsp):

    feature_comb_count = 0
    for record in param_feature_comb_list:
        feature_comb_count += len(record)

    run_count = 0
    ret_total_score_list = []
    next_complete = 0
    for temp in param_feature_comb_list:
        for feature_comb_list in temp:
            run_count += 1
            prev_complete = run_count / feature_comb_count * 100
            prev_complete = round(prev_complete)
            if prev_complete != next_complete:
                print("(", param_chosen_ml, "), (", param_size, "), (",
                      param_dsp, "), complete: ", prev_complete, '%')
                next_complete = prev_complete

            clustering_normal_feature_array = get_chosen_field_array(feature_comb_list,
                                                                     param_clustering_normal_feature_array)
            clustering_attack_feature_array = get_chosen_field_array(feature_comb_list,
                                                                     param_clustering_attack_feature_array)
            training_normal_feature_array = get_chosen_field_array(feature_comb_list,
                                                                   param_training_normal_feature_array)
            training_attack_feature_array = get_chosen_field_array(feature_comb_list,
                                                                   param_training_attack_feature_array)
            testing_normal_feature_array = get_chosen_field_array(feature_comb_list,
                                                                  param_testing_normal_feature_array)
            testing_attack_feature_array = get_chosen_field_array(feature_comb_list,
                                                                  param_testing_attack_feature_array)

            ml_score_list = ml_select(clustering_normal_feature_array, param_clustering_normal_label_list,
                                      clustering_attack_feature_array, param_clustering_attack_label_list,
                                      training_normal_feature_array, param_training_normal_label_list,
                                      training_attack_feature_array, param_training_attack_label_list,
                                      testing_normal_feature_array, param_testing_normal_label_list,
                                      testing_attack_feature_array, param_testing_attack_label_list,
                                      param_chosen_ml, param_axis_value_list, feature_comb_list)

            ml_score_list.append(feature_comb_list)
            ml_score_list.append(param_axis_value_list)
            ml_score_list.append(param_dsp)
            ml_score_list.append(param_size)
            ret_total_score_list.append(ml_score_list)

    return ret_total_score_list


def get_mixed_label_array(param_normal_label_list, param_attack_label_list):
    mixed_label_list = copy.deepcopy(param_normal_label_list)
    for i in param_attack_label_list:
        mixed_label_list.append(i)

    ret_mixed_label_array = np.array(mixed_label_list)
    return ret_mixed_label_array


def get_weighted_f1_avg_score(param_array, param_predicted_label):
    class_report = classification_report(param_array, param_predicted_label, output_dict=True)
    weighted_avg = class_report['weighted avg']
    f1_score = weighted_avg['f1-score']

    return f1_score
