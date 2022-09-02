import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(1, './ml')
from constant import ml
from utility import feature_management, function
import warnings

warnings.filterwarnings("ignore")


def get_violin_plot_list(param_feature_array):
    ret_violin_plot_list = []

    for index in range(0, param_feature_array.shape[1]):
        feature_field_list = param_feature_array[:, index:index + 1]
        feature_type_list = []
        for feature_field in feature_field_list:
            feature_type_list.append(feature_field[0])

        ret_violin_plot_list.append(feature_type_list)

    return ret_violin_plot_list


def draw_dataset_scope(param_normal_feature_array, param_attack_feature_array, param_title):
    violin_plot_list_1 = get_violin_plot_list(param_normal_feature_array)
    violin_plot_list_2 = get_violin_plot_list(param_attack_feature_array)
    plt.clf()
    plt.style.use('default')
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.figsize'] = (16, 10)

    fig, ax = plt.subplots()
    ax.violinplot(violin_plot_list_1, showmeans=True)
    ax.violinplot(violin_plot_list_2, showmeans=True)
    ax.set_xticklabels(ml.feature_types)
    xtick_list = [i for i in range(1, len(ml.feature_types) + 1)]
    ax.set_xticks(xtick_list)
    ax.set_xlabel("Feature Type")
    ax.set_ylabel('Value')
    plt.title(param_title)
    plt.savefig('.//feature_info//' + param_title + '.png', dpi=100, bbox_inches='tight')


def draw_dataset_size(param_ratio, param_label, param_title):
    plt.clf()
    plt.pie(param_ratio, labels=param_label, autopct='%.1f%%')
    plt.title(param_title)

    plt.savefig('.//feature_info//' + param_title + '.png', dpi=100, bbox_inches='tight')