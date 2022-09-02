class Constant:
    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise Exception('impossible to assign + ' + name + '.')
        self.__dict__[name] = value

    def __delattr__(self, name):
        if name in self.__dict__:
            raise Exception('impossible to delete ' + name + '.')

    def __init__(self):
        self.benign = 0
        self.malignant = 1

        self.k_means = 'k_means'
        self.k_means_option_1_list = ['elkan', 'auto', 'full']
        self.kmeans_axis = [len(self.k_means_option_1_list)]

        self.ada_boost = 'ada_boost'
        self.ada_boost_option_1_list = ['SAMME', 'SAMME.R']
        self.ada_boost_axis = [len(self.ada_boost_option_1_list)]

        self.db_scan = 'db_scan'
        self.db_scan_option_1_list = ['auto', 'ball_tree', 'kd_tree', 'brute']
        self.db_scan_axis = [len(self.db_scan_option_1_list)]

        self.decision_tree = 'decision_tree'
        self.decision_tree_option_1_list = ['gini', 'entropy']
        self.decision_tree_option_2_list = ['best', 'random']
        self.decision_tree_option_3_list = ['auto', 'sqrt', 'log2']
        self.decision_tree_axis = [len(self.decision_tree_option_1_list), len(self.decision_tree_option_2_list),
                                   len(self.decision_tree_option_3_list)]

        self.gaussian_mixture = 'gaussian_mixture'
        self.gaussian_mixture_option_1_list = ['full', 'tied', 'diag', 'spherical']
        self.gaussian_mixture_option_2_list = [True, False]
        self.gaussian_mixture_axis = [len(self.gaussian_mixture_option_1_list),
                                      len(self.gaussian_mixture_option_2_list)]

        self.gradient_boost = 'gradient_boost'
        self.gradient_boost_option_1_list = ['deviance', 'exponential']
        self.gradient_boost_option_2_list = [True, False]
        self.gradient_boost_option_3_list = ['auto', 'sqrt', 'log2', None]
        self.gradient_boost_axis = [len(self.gradient_boost_option_1_list), len(self.gradient_boost_option_2_list),
                                    len(self.gradient_boost_option_3_list)]

        self.hierarchy = 'hierarchy '
        self.hierarchy_axis = [1]

        self.knn = 'k_nearest_neighbor'
        self.knn_option_1_list = ['uniform', 'distance']
        self.knn_option_2_list = ['auto', 'ball_tree', 'kd_tree', 'brute']
        self.knn_axis = [len(self.knn_option_1_list), len(self.knn_option_2_list)]

        self.perceptron = 'perceptron'
        self.perceptron_option_1_list = [True, False]
        self.perceptron_axis = [len(self.perceptron_option_1_list)]

        self.random_forest = 'random_forest'
        self.random_forest_option_1_list = ['gini', 'entropy']
        self.random_forest_option_2_list = ['sqrt', 'log2', None]
        self.random_forest_option_3_list = ['balanced', 'balanced_subsample']
        self.random_forest_axis = [len(self.random_forest_option_1_list), len(self.random_forest_option_2_list),
                                   len(self.random_forest_option_3_list)]

        self.logistic_regression = 'logistic_regression'
        self.logistic_regression_option_1_list = ['l2', 'none']
        self.logistic_regression_option_2_list = [True, False]
        self.logistic_regression_option_3_list = ['newton-cg', 'lbfgs', 'sag']
        self.logistic_regression_axis = [len(self.logistic_regression_option_1_list),
                                         len(self.logistic_regression_option_2_list),
                                         len(self.logistic_regression_option_3_list)]

        self.gaussian_nb = 'gaussian_naive_bayes'
        self.gaussian_nb_option_1_list = [None, None]
        self.gaussian_nb_axis = [len(self.gaussian_nb_option_1_list)]

        self.svm = 'svm'
        self.svm_option_1_list = ['linear', 'poly', 'rbf', 'sigmoid']
        self.svm_option_2_list = ['scale', 'auto']
        self.svm_option_3_list = ['ovo', 'ovr']
        self.svm_axis = [len(self.svm_option_1_list), len(self.svm_option_2_list), len(self.svm_option_3_list)]

        self.ml_name_list = [self.k_means, self.ada_boost, self.db_scan, self.decision_tree, self.gaussian_mixture,
                             self.gradient_boost, self.hierarchy, self.knn, self.perceptron, self.random_forest,
                             self.logistic_regression, self.gaussian_nb, self.svm]

        self.ml_type = {self.k_means: [1], self.ada_boost: [1], self.db_scan: [1], self.decision_tree: [1],
                        self.gaussian_mixture: [1], self.gradient_boost: [1], self.hierarchy: [1], self.knn: [1],
                        self.random_forest: [1], self.logistic_regression: [1],
                        self.gaussian_nb: [1], self.svm: [1], self.perceptron: [1]}  # XXX_axis  self.perceptron: [1]

        # self.ml_type = {self.ada_boost: [1], self.k_means: [1]}

        self.feature_types = ['TCT', 'DCTAC', 'DCTBD', 'KWD', 'WPM', 'KWR', 'MR', 'MA', 'MAC', 'MBD', 'PR', 'RAC']


import sys

sys.modules[__name__] = Constant()
