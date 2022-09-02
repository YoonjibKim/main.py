import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import to_categorical
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

    n_feat = mixed_training_feature_array.shape[1]
    n_class = len(set(mixed_training_label_array))
    epo = 10

    model = Sequential()
    model.add(Dense(20, input_dim=n_feat))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(n_class))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    onehot_mixed_training_label_array = to_categorical(mixed_training_label_array)

    model.fit(mixed_training_feature_array, onehot_mixed_training_label_array, epochs=epo, batch_size=10)

    dnn_pred = model.predict(mixed_testing_feature_array)
    dnn_argmax_pred = np.argmax(dnn_pred, axis=1)

    score = ml_algorithm.get_weighted_f1_avg_score(dnn_argmax_pred, mixed_testing_label_array)

    return score
