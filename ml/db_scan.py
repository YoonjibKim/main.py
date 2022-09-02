import sys
sys.path.insert(1, './ml')
from ml import ml_algorithm
import numpy as np
from sklearn.cluster import DBSCAN


def run(param_clustering_normal_feature_array, param_clustering_normal_label_list,
        param_clustering_attack_feature_array, param_clustering_attack_label_list, param_axis_value_list):

    mixed_clustering_label_array = ml_algorithm.get_mixed_label_array(param_clustering_normal_label_list,
                                                           param_clustering_attack_label_list)
    mixed_clustering_feature_array = np.append(param_clustering_normal_feature_array,
                                               param_clustering_attack_feature_array, axis=0)

    dbs = DBSCAN(eps=200000)
    dbs.fit(mixed_clustering_feature_array)
    label_dbs = dbs.labels_

    score = ml_algorithm.get_weighted_f1_avg_score(mixed_clustering_label_array, label_dbs)

    return score
