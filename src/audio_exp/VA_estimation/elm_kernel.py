import numpy as np
from sklearn.metrics import pairwise

class ELM:
    def __init__(self, c=1, weighted=False, kernel='rbf', deg = 3, is_classification = False):
        super(self.__class__, self).__init__()

        assert kernel in ["rbf", "linear", "poly", "sigmoid"]
        self.x_train = []
        self.C = c
        self.weighted = weighted
        self.beta = []
        self.kernel = kernel
        self.is_classification=is_classification
        self.deg = deg

    def fit(self, x_train, y_train):
        """
        Calculate beta using kernel.
        :param x_train: features of train set
        :param y_train: labels of train set
        :return:
        """
        self.x_train = x_train

        if self.is_classification:
            class_num = 4
            n = len(x_train)
            y_one_hot = np.eye(class_num)[y_train]
        
        else:
            n = len(x_train)
            y_one_hot = y_train

        if self.kernel == 'rbf':
            kernel_func = pairwise.rbf_kernel(x_train)
        elif self.kernel == 'poly':
            kernel_func = pairwise.polynomial_kernel(x_train, degree = self.deg)
        elif self.kernel == 'sigmoid':
            kernel_func = pairwise.sigmoid_kernel(x_train)
        elif self.kernel == 'linear':
            kernel_func = pairwise.linear_kernel(x_train)

        if self.is_classification and self.weighted:
            W = np.zeros((n, n))
            hist = np.zeros(class_num)
            for label in y_train:
                hist[label] += 1
            hist = 1 / hist
            for i in range(len(y_train)):
                W[i, i] = hist[y_train[i]]
                
           
            beta = np.matmul(np.linalg.inv(np.matmul(W, kernel_func) +
                                           np.identity(n) / np.float(self.C)), np.matmul(W, y_one_hot))
        else:
            # numpy 版本兼容问题解决 使用np.float64
            beta = np.matmul(np.linalg.inv(kernel_func + np.identity(n) / np.float32(self.C)), y_one_hot)
        self.beta = beta

    def predict(self, x_test):
        """
        Predict label probabilities of new data using calculated beta.
        :param x_test: features of new data
        :return: class probabilities of new data
        """

        if self.kernel == 'rbf':
            kernel_func = pairwise.rbf_kernel(x_test, self.x_train)
        elif self.kernel == 'poly':
            kernel_func = pairwise.polynomial_kernel(x_test, self.x_train)
        elif self.kernel == 'sigmoid':
            kernel_func = pairwise.sigmoid_kernel(x_test, self.x_train)
        elif self.kernel == 'linear':
            kernel_func = pairwise.linear_kernel(x_test, self.x_train)
        pred = np.matmul(kernel_func, self.beta)
        return pred

