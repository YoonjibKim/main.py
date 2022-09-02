from constant import common, ml
from ml import ml_algorithm
import numpy as np
from utility.function import read_json, feature_preprocessing_acn_data, save_feature_type_to_csv, save_score_to_csv, \
    save_mean_std_to_csv, save_feature_span_to_csv
from utility.feature_management import feature_mean_std, get_feature_span, generate_features
from utility import visualize_dataset
from itertools import product, combinations


def parameter_setting():
    # ---------------------------------------------------------------------------------------
    feature_comb_from = 0  # 0 ~ 12
    feature_comb_to = 12
    normal_feature_gen_start_span = 0  # beginning ratio of normal feature data
    normal_feature_gen_end_span = 2  # n: n times of avg + std
    total_data_size = 500  # testing total data size (normal + attack)
    testing_dataset_percent = 0.3  # percent 1.0: 100 %
    feature_data_step = 1  # data size variation (deprecated)
    attack_feature_dsp = 10  # n: n times of shifting dataset
    # ---------------------------------------------------------------------------------------

    attack_feature_gen_start_span_list = []
    attack_feature_gen_end_span_list = []
    for distance in range(1, attack_feature_dsp + 1):
        feature_span = normal_feature_gen_end_span - normal_feature_gen_start_span
        start = feature_span * distance
        end = start + feature_span
        attack_feature_gen_start_span_list.append(start)
        attack_feature_gen_end_span_list.append(end)

    ret_feature_comb_count_list = [feature_comb_from, feature_comb_to]
    ret_normal_feature_gen_span_list = [[normal_feature_gen_start_span for i in range(attack_feature_dsp)],
                                        [normal_feature_gen_end_span for i in range(attack_feature_dsp)]]
    ret_attack_feature_gen_span_list = [attack_feature_gen_start_span_list, attack_feature_gen_end_span_list]

    clustering_normal_size = round(total_data_size * 0.5)
    clustering_attack_size = round(total_data_size * 0.5)

    testing_dataset_size = round(total_data_size * testing_dataset_percent)
    testing_normal_size = round(testing_dataset_size * 0.5)
    testing_attack_size = round(testing_dataset_size * 0.5)

    training_normal_size = round((total_data_size - testing_dataset_size) * 0.5)
    training_attack_size = round((total_data_size - testing_dataset_size) * 0.5)

    data_size = [clustering_normal_size, clustering_attack_size, training_normal_size, training_attack_size,
                 testing_normal_size, testing_attack_size, total_data_size]

    ret_feature_size_step_dsp_list = [data_size, feature_data_step, attack_feature_dsp]

    return ret_feature_comb_count_list, ret_normal_feature_gen_span_list, ret_attack_feature_gen_span_list, \
           ret_feature_size_step_dsp_list


if __name__ == '__main__':
    print('program start')

    raw_dataset = read_json(common.dataset_path)
    raw_normal_feature_list, raw_feature_label_list = feature_preprocessing_acn_data(raw_dataset)
    feature_mean_list, feature_std_list = feature_mean_std(raw_normal_feature_list)
    feature_comb_count_list, normal_feature_gen_span_list, attack_feature_gen_span_list, feature_size_step_dsp_list \
        = parameter_setting()

    begin_span_normal_feature_list, end_span_normal_feature_list = get_feature_span(feature_mean_list,
                                                                                    feature_std_list,
                                                                                    normal_feature_gen_span_list)
    begin_span_attack_feature_list, end_span_attack_feature_list = get_feature_span(feature_mean_list,
                                                                                    feature_std_list,
                                                                                    attack_feature_gen_span_list)

    save_feature_span_to_csv(begin_span_normal_feature_list, end_span_normal_feature_list, 'normal')
    save_feature_span_to_csv(begin_span_attack_feature_list, end_span_attack_feature_list, 'attack')

    clustering_normal_size_dsp_feature_array = generate_features(begin_span_normal_feature_list,
                                                                 end_span_normal_feature_list,
                                                                 feature_size_step_dsp_list[0][0],
                                                                 feature_comb_count_list[1]-feature_comb_count_list[0],
                                                                 feature_size_step_dsp_list[2])

    training_normal_size_dsp_feature_array = generate_features(begin_span_normal_feature_list,
                                                               end_span_normal_feature_list,
                                                               feature_size_step_dsp_list[0][2],
                                                               feature_comb_count_list[1]-feature_comb_count_list[0], 1)

    testing_normal_size_dsp_feature_array = generate_features(begin_span_normal_feature_list,
                                                              end_span_normal_feature_list,
                                                              feature_size_step_dsp_list[0][4],
                                                              feature_comb_count_list[1]-feature_comb_count_list[0],
                                                              feature_size_step_dsp_list[2])

    clustering_attack_size_dsp_feature_array = generate_features(begin_span_attack_feature_list,
                                                                 end_span_attack_feature_list,
                                                                 feature_size_step_dsp_list[0][1],
                                                                 feature_comb_count_list[1]-feature_comb_count_list[0],
                                                                 feature_size_step_dsp_list[2])

    training_attack_size_dsp_feature_array = generate_features(begin_span_attack_feature_list,
                                                               end_span_attack_feature_list,
                                                               feature_size_step_dsp_list[0][3],
                                                               feature_comb_count_list[1]-feature_comb_count_list[0], 1)

    testing_attack_size_dsp_feature_array = generate_features(begin_span_attack_feature_list,
                                                              end_span_attack_feature_list,
                                                              feature_size_step_dsp_list[0][5],
                                                              feature_comb_count_list[1]-feature_comb_count_list[0],
                                                              feature_size_step_dsp_list[2])

    size = feature_size_step_dsp_list[0][6]
    dsp = feature_size_step_dsp_list[2]
    flag = True
    temp_clustering_normal_size_dsp_feature_list = []
    temp_training_normal_size_dsp_feature_list = []
    temp_testing_normal_size_dsp_feature_list = []
    temp_clustering_attack_size_dsp_feature_list = []
    temp_training_attack_size_dsp_feature_list = []
    temp_testing_attack_size_dsp_feature_list = []

    for feature_dsp in range(0, dsp):
        for feature_size in range(0, feature_size_step_dsp_list[0][0]):
            temp = clustering_normal_size_dsp_feature_array[feature_size][feature_dsp]
            temp_clustering_normal_size_dsp_feature_list.append(temp)

        for feature_size in range(0, feature_size_step_dsp_list[0][2]):
            temp = training_normal_size_dsp_feature_array[feature_size][0]
            temp_training_normal_size_dsp_feature_list.append(temp)

        for feature_size in range(0, feature_size_step_dsp_list[0][4]):
            temp = testing_normal_size_dsp_feature_array[feature_size][feature_dsp]
            temp_testing_normal_size_dsp_feature_list.append(temp)

        for feature_size in range(0, feature_size_step_dsp_list[0][1]):
            temp = clustering_attack_size_dsp_feature_array[feature_size][feature_dsp]
            temp_clustering_attack_size_dsp_feature_list.append(temp)

        for feature_size in range(0, feature_size_step_dsp_list[0][3]):
            temp = training_attack_size_dsp_feature_array[feature_size][0]
            temp_training_attack_size_dsp_feature_list.append(temp)

        for feature_size in range(0, feature_size_step_dsp_list[0][5]):
            temp = testing_attack_size_dsp_feature_array[feature_size][feature_dsp]
            temp_testing_attack_size_dsp_feature_list.append(temp)

        temp_clustering_normal_size_dsp_feature_array = np.array(temp_clustering_normal_size_dsp_feature_list)
        temp_training_normal_size_dsp_feature_array = np.array(temp_training_normal_size_dsp_feature_list)
        temp_testing_normal_size_dsp_feature_array = np.array(temp_testing_normal_size_dsp_feature_list)
        temp_clustering_attack_size_dsp_feature_array = np.array(temp_clustering_attack_size_dsp_feature_list)
        temp_training_attack_size_dsp_feature_array = np.array(temp_training_attack_size_dsp_feature_list)
        temp_testing_attack_size_dsp_feature_array = np.array(temp_testing_attack_size_dsp_feature_list)

        temp_clustering_normal_size_dsp_feature_list.clear()
        temp_training_normal_size_dsp_feature_list.clear()
        temp_testing_normal_size_dsp_feature_list.clear()
        temp_clustering_attack_size_dsp_feature_list.clear()
        temp_training_attack_size_dsp_feature_list.clear()
        temp_testing_attack_size_dsp_feature_list.clear()

        visualize_dataset.draw_dataset_scope(temp_clustering_normal_size_dsp_feature_array,
                                             temp_clustering_attack_size_dsp_feature_array,
                                             'clustering_' + str(size) + '_' + str(feature_dsp))

        visualize_dataset.draw_dataset_scope(temp_training_normal_size_dsp_feature_array,
                                             temp_training_attack_size_dsp_feature_array,
                                             'training_' + str(size) + '_' + str(feature_dsp))

        visualize_dataset.draw_dataset_scope(temp_testing_normal_size_dsp_feature_array,
                                             temp_testing_attack_size_dsp_feature_array,
                                             'testing_' + str(size) + '_' + str(feature_dsp))

        if flag is True:
            flag = False
            ratio = [temp_clustering_normal_size_dsp_feature_array.shape[0],
                     temp_clustering_attack_size_dsp_feature_array.shape[0]]
            label = ['Normal Data Count: ' + str(temp_clustering_normal_size_dsp_feature_array.shape[0]),
                     'Attack Data Count: ' + str(temp_clustering_attack_size_dsp_feature_array.shape[0])]
            visualize_dataset.draw_dataset_size(ratio, label, 'Total_Clustering_Data_Count_' +
                                                str(temp_clustering_normal_size_dsp_feature_array.shape[0] +
                                                    temp_clustering_attack_size_dsp_feature_array.shape[0]))

            ratio = [temp_training_normal_size_dsp_feature_array.shape[0],
                     temp_training_attack_size_dsp_feature_array.shape[0]]
            label = ['Training Normal Data Count: ' + str(temp_training_normal_size_dsp_feature_array.shape[0]),
                     'Training Attack Data Count: ' + str(temp_training_attack_size_dsp_feature_array.shape[0])]
            visualize_dataset.draw_dataset_size(ratio, label, 'Total_Training_Data_Count_' +
                                                str(temp_training_normal_size_dsp_feature_array.shape[0] +
                                                    temp_training_attack_size_dsp_feature_array.shape[0]))

            ratio = [temp_testing_normal_size_dsp_feature_array.shape[0],
                     temp_testing_attack_size_dsp_feature_array.shape[0]]
            label = ['Testing Normal Data Count: ' + str(temp_testing_normal_size_dsp_feature_array.shape[0]),
                     'Testing Attack Data Count: ' + str(temp_testing_attack_size_dsp_feature_array.shape[0])]
            visualize_dataset.draw_dataset_size(ratio, label, 'Total_Testing_Data_Count_' +
                                                str(temp_testing_normal_size_dsp_feature_array.shape[0] +
                                                    temp_testing_attack_size_dsp_feature_array.shape[0]))

            ratio = [temp_training_normal_size_dsp_feature_array.shape[0] +
                     temp_training_attack_size_dsp_feature_array.shape[0],
                     temp_testing_normal_size_dsp_feature_array.shape[0] +
                     temp_testing_attack_size_dsp_feature_array.shape[0]]
            label = ['Total Training Data Count: ' + str(temp_training_normal_size_dsp_feature_array.shape[0] +
                                                         temp_training_attack_size_dsp_feature_array.shape[0]),
                     'Total Testing Data Count: ' + str(temp_testing_normal_size_dsp_feature_array.shape[0] +
                                                        temp_testing_attack_size_dsp_feature_array.shape[0])]
            visualize_dataset.draw_dataset_size(ratio, label, 'Total_Training_and_Testing_Data_Count_' +
                                                str(ratio[0] + ratio[1]))

    clustering_normal_label_list = [0 for i in range(feature_size_step_dsp_list[0][0])]
    training_normal_label_list = [0 for i in range(feature_size_step_dsp_list[0][2])]
    testing_normal_label_list = [0 for i in range(feature_size_step_dsp_list[0][4])]

    clustering_attack_label_list = [1 for i in range(feature_size_step_dsp_list[0][1])]
    training_attack_label_list = [1 for i in range(feature_size_step_dsp_list[0][3])]
    testing_attack_label_list = [1 for i in range(feature_size_step_dsp_list[0][5])]

    total_ml_score_list = []
    feature_comb_list = []
    for feature_step in range(feature_comb_count_list[0], feature_comb_count_list[1]):
        temp_list = []
        for comb_step in list(combinations(ml.feature_types, feature_step + 1)):
            temp_list.append(comb_step)
        feature_comb_list.append(temp_list)
    save_feature_type_to_csv(feature_comb_list)

    for chosen_ml in ml.ml_type:
        temp_value_list = []
        axis_size_list = ml.ml_type[chosen_ml]

        for axis_size in axis_size_list:
            temp_list = []
            for axis_value in range(0, axis_size):
                temp_list.append(axis_value)
            temp_value_list.append(temp_list)

        permutation = list(product(*temp_value_list))
        each_ml_score_list = []
        for axis_value_list in permutation:
            for dsp in range(0, feature_size_step_dsp_list[2]):

                clustering_normal_size_dsp_feature_list = []
                training_normal_size_dsp_feature_list = []
                testing_normal_size_dsp_feature_list = []

                clustering_attack_size_dsp_feature_list = []
                training_attack_size_dsp_feature_list = []
                testing_attack_size_dsp_feature_list = []

                for _size in range(0, feature_size_step_dsp_list[0][0]):
                    clustering_normal_size_dsp_feature_list.append(clustering_normal_size_dsp_feature_array[_size][dsp])
                for _size in range(0, feature_size_step_dsp_list[0][2]):
                    training_normal_size_dsp_feature_list.append(training_normal_size_dsp_feature_array[_size][0])
                for _size in range(0, feature_size_step_dsp_list[0][4]):
                    testing_normal_size_dsp_feature_list.append(testing_normal_size_dsp_feature_array[_size][dsp])
                for _size in range(0, feature_size_step_dsp_list[0][1]):
                    clustering_attack_size_dsp_feature_list.append(clustering_attack_size_dsp_feature_array[_size][dsp])
                for _size in range(0, feature_size_step_dsp_list[0][3]):
                    training_attack_size_dsp_feature_list.append(training_attack_size_dsp_feature_array[_size][0])
                for _size in range(0, feature_size_step_dsp_list[0][5]):
                    testing_attack_size_dsp_feature_list.append(testing_attack_size_dsp_feature_array[_size][dsp])

                score_list = ml_algorithm.run_score(chosen_ml, axis_value_list,
                                                    clustering_normal_size_dsp_feature_list,
                                                    clustering_normal_label_list,
                                                    training_normal_size_dsp_feature_list,
                                                    training_normal_label_list,
                                                    testing_normal_size_dsp_feature_list,
                                                    testing_normal_label_list,
                                                    clustering_attack_size_dsp_feature_list,
                                                    clustering_attack_label_list,
                                                    training_attack_size_dsp_feature_list,
                                                    training_attack_label_list,
                                                    testing_attack_size_dsp_feature_list,
                                                    testing_attack_label_list,
                                                    feature_comb_list, size, dsp)

                score_list.append(feature_size_step_dsp_list[2])
                each_ml_score_list.append(score_list)

        save_score_to_csv(each_ml_score_list, chosen_ml + '_all_scores')

    print('program end')
