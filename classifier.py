from sklearn.svm import OneClassSVM
import itertools


class Classifier:
    def __init__(self, type_, for_global_option=False, for_goal_option=False, env_termination_checker=None):
        assert type_ in ["initiation", "termination"]

        # it cant be both global and goal
        assert not (for_global_option and for_goal_option)

        self.type_ = type_
        self.for_global_option = for_global_option
        self.for_goal_option = for_goal_option

        self.env_termination_checker = env_termination_checker
        self.one_class_svm = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")

        # if self is goal or global option, termination checker should be provided
        if self.for_global_option or self.for_goal_option:
            assert self.env_termination_checker is not None
        else:
            assert self.env_termination_checker is None

    def check(self, x):
        if self.type_ == "initiation" and self.for_global_option:
            return True

        if self.type_ == "termination" and (self.for_global_option or self.for_goal_option):
            return self.env_termination_checker(x)

        # TODO: implement
        self.one_class_svm.predict(x)
        return True

    def train_one_class(self, x):
        assert self.type_ == "initiation"

        x = list(itertools.chain(*x))
        self.one_class_svm.fit(x)
