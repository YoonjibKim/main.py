import random
import numpy as np


def feature_mean_std(param_feature_list):
    feature_mean_list = []
    feature_std_list = []
    feature_array = np.array(param_feature_list)

    for i in feature_array.T:
        feature_mean_list.append(np.mean(i))
        feature_std_list.append(np.std(i))

    return feature_mean_list, feature_std_list


def get_feature_span(param_feature_mean_list, param_feature_std_list, param_feature_gen_span_list):
    ret_begin_span_feature_list = []
    ret_end_span_feature_list = []

    for param_feature_mean, param_feature_std in zip(param_feature_mean_list, param_feature_std_list):
        base_span = param_feature_mean + param_feature_std

        feature_type_start_span_list = []
        for step in param_feature_gen_span_list[0]:
            start_span = base_span * step
            feature_type_start_span_list.append(start_span)

        feature_type_end_span_list = []
        for step in param_feature_gen_span_list[1]:
            end_span = base_span * step
            feature_type_end_span_list.append(end_span)

        ret_begin_span_feature_list.append(feature_type_start_span_list)
        ret_end_span_feature_list.append(feature_type_end_span_list)

    return ret_begin_span_feature_list, ret_end_span_feature_list


def generate_features(param_begin_span_list, param_end_span_list, param_data_size, param_feature_count, param_dsp):
    total_feature_dsp_list = []

    for feature_count in range(0, param_feature_count):
        begin_dsp_list = param_begin_span_list[feature_count]
        end_dsp_list = param_end_span_list[feature_count]
        feature_dsp_list = []
        for dsp_index in range(0, param_dsp):
            temp_list = np.random.uniform(begin_dsp_list[dsp_index], end_dsp_list[dsp_index], param_data_size)
            round_list = []
            for element in temp_list:
                round_list.append(round(element))
            feature_dsp_list.append(round_list)

        total_feature_dsp_list.append(feature_dsp_list)

    return np.array(total_feature_dsp_list).T
