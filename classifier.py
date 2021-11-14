from sklearn.svm import OneClassSVM
import itertools


class Classifier:
    def __init__(self, goal_or_global=False, env_termination_checker=None):
        self.goal_or_global = goal_or_global
        self.env_termination_checker = env_termination_checker
        self.one_class_svm = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")

        if self.goal_or_global:
            assert self.env_termination_checker is not None
        else:
            assert self.env_termination_checker is None

    def check(self, x):
        if self.goal_or_global:
            return self.env_termination_checker(x)

        # TODO: implement
        return True

    def train_one_class(self, x):
        x = list(itertools.chain(*x))
        self.one_class_svm.fit(x)
