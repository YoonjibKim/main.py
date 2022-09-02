import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import sys

sys.path.insert(1, './ml')
from ml import ml_algorithm


def run(param_training_normal_feature_array, param_training_normal_label_list,
        param_training_attack_feature_array, param_training_attack_label_list,
        param_testing_normal_feature_array, param_testing_normal_label_list,
        param_testing_attack_feature_array, param_testing_attack_label_list, param_axis_value_list):
    mixed_training_label_array = ml_algorithm.get_mixed_label_array(param_training_normal_label_list,
                                                                    param_training_attack_label_list)
    mixed_training_feature_array = np.append(param_training_normal_feature_array, param_training_attack_feature_array,
                                             axis=0)
    mixed_testing_label_array = ml_algorithm.get_mixed_label_array(param_testing_normal_label_list,
                                                                   param_testing_attack_label_list)
    mixed_testing_feature_array = np.append(param_testing_normal_feature_array, param_testing_attack_feature_array,
                                            axis=0)

    std_scale = StandardScaler()
    std_scale.fit(mixed_training_feature_array)
    X_tn_std = std_scale.transform(mixed_training_feature_array)
    X_te_std = std_scale.transform(mixed_testing_feature_array)

    clf_gbt = GradientBoostingClassifier(max_depth=2, learning_rate=0.01, random_state=0)
    clf_gbt.fit(X_tn_std, mixed_training_label_array)

    pred_gboost = clf_gbt.predict(X_te_std)

    score = ml_algorithm.get_weighted_f1_avg_score(mixed_testing_label_array, pred_gboost)

    return score
