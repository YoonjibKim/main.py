import csv
from datetime import datetime
import json
import os
import shutil
import numpy as np
import sys

sys.path.insert(1, './constant')
from constant import ml


def read_json(path):
    with open(path, 'r') as json_file:
        json_data = json.load(json_file)

    return json_data


def change_boolean_to_number(bool_param):
    number = 0
    if bool_param is True:
        number = 1

    return number


def null_check(objects, main_field_names, sub_field_name):
    check_fields = True

    for main_field_name in main_field_names:
        if objects[main_field_name] is None:
            check_fields = False
            break

        if main_field_name == sub_field_name:
            for sub_object in objects[sub_field_name]:
                if sub_object is None:
                    check_fields = False
                    break

    return check_fields


def time_diff_sec_acn_data(posterior_date, prior_date):
    str1 = '{0}'.format(prior_date).rstrip(' GMT')
    str2 = '{0}'.format(posterior_date).rstrip(' GMT')

    dts1 = datetime.strptime(str1, '%a, %d %b %Y %X')
    dts2 = datetime.strptime(str2, '%a, %d %b %Y %X')

    diff = ((dts2 - dts1).days * 86400) + (dts2 - dts1).seconds
    return diff


def feature_preprocessing_acn_data(_items):
    items = _items['_items']
    authentication_information_list = []
    main_field_names = ['_id', 'doneChargingTime', 'disconnectTime', 'connectionTime', 'kWhDelivered', 'userInputs']

    feature_label_list = []
    for item in items:
        if null_check(item, main_field_names, 'userInputs'):  # true: not null
            feature_label_list.append(ml.benign)
            sub_fields = item['userInputs'][0]
            not_bool_paymentRequired = change_boolean_to_number(sub_fields['paymentRequired'])

            temp_list = [abs(time_diff_sec_acn_data(item['disconnectTime'], item['connectionTime'])),
                         abs(time_diff_sec_acn_data(item['doneChargingTime'], item['connectionTime'])),
                         abs(time_diff_sec_acn_data(item['disconnectTime'], item['doneChargingTime'])),
                         round(item['kWhDelivered']), sub_fields['WhPerMile'], round(sub_fields['kWhRequested']),
                         sub_fields['milesRequested'], sub_fields['minutesAvailable'],
                         abs(time_diff_sec_acn_data(sub_fields['modifiedAt'], item['connectionTime'])),
                         abs(time_diff_sec_acn_data(item['disconnectTime'], sub_fields['modifiedAt'])),
                         not_bool_paymentRequired, time_diff_sec_acn_data(sub_fields['requestedDeparture'],
                                                                          item['connectionTime'])]

            authentication_information_list.append(temp_list)

    return authentication_information_list, feature_label_list


def save_feature_type_to_csv(param_data_double_list):
    with open('.//feature_info//feature_combination.csv', 'w', newline='') as f:
        write = csv.writer(f)

        for single_list in param_data_double_list:
            write.writerows(single_list)


def save_score_to_csv(param_score_list, param_info):
    score_list = []
    feature_combination_list = []
    feature_size_list = []
    feature_dsp_list = []

    for outer_score_list in param_score_list:
        for index in range(0, len(outer_score_list)-1):
            inner_score_list = outer_score_list[index]
            score_list.append(inner_score_list[1])
            feature_combination_list.append(inner_score_list[2])
            feature_size_list.append(inner_score_list[4])
            feature_dsp_list.append(inner_score_list[5])

    with open('.//score_result//' + param_info + '.csv', 'w', newline='') as f:
        write = csv.writer(f)

        for score, feature_comb, feature_size, feature_dsp in zip(score_list, feature_combination_list,
                                                           feature_size_list, feature_dsp_list):
            write.writerow([str(score), feature_comb ,str(feature_dsp), str(feature_size)])


def save_mean_std_to_csv(param_mean_list, param_std_list):
    with open('.//feature_info//mean_std.csv', 'w', newline='') as f:
        write = csv.writer(f)
        write.writerow(['mean'])
        write.writerow(param_mean_list)
        write.writerow(['std'])
        write.writerow(param_std_list)


def save_feature_span_to_csv(param_bottom_span_list, param_top_span_list, additional_file_name):
    with open('.//feature_info//' + additional_file_name + '_feature_span.csv', 'w', newline='') as f:
        write = csv.writer(f)

        write.writerow(['begin'])
        for index in range(0, len(ml.feature_types)):
            write.writerow([ml.feature_types[index]])
            write.writerow(param_bottom_span_list[index])

        write.writerow([''])

        write.writerow(['end'])
        for index in range(0, len(ml.feature_types)):
            write.writerow([ml.feature_types[index]])
            write.writerow(param_top_span_list[index])