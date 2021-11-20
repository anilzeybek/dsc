import random
from copy import deepcopy
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC


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
        self.two_class_svm = SVC(kernel="rbf", gamma="scale", class_weight="balanced")
        self.one_class_trained = False
        self.two_class_trained = False

        # if self is goal or global option, termination checker should be provided
        if self.type_ == "termination" and (self.for_global_option or self.for_goal_option):
            assert self.env_termination_checker is not None
        elif self.type_ == "termination":
            assert self.env_termination_checker is None

    def check(self, x) -> bool:
        if self.type_ == "initiation" and self.for_global_option:
            return True

        if self.type_ == "termination" and (self.for_global_option or self.for_goal_option):
            return self.env_termination_checker(x)

        if self.two_class_trained:
            return self.two_class_svm.predict([x])[0] == 1

        return self.one_class_svm.predict([x])[0] == 1

    def sample(self):
        # sampling only valid for termination classifiers
        assert self.type_ == "termination"
        # at least one_class must be trained
        assert self.one_class_trained

        if self.two_class_trained:
            # TODO:
            return ...

        return random.sample(self.one_class_train_examples)

    def train_one_class(self, xs):
        assert self.type_ == "initiation"
        assert not self.for_global_option
        assert not self.one_class_trained

        self.one_class_train_examples = deepcopy(xs)

        # TODO: it may be fitting wrong, check when find true termination function
        self.one_class_svm.fit(xs)
        self.one_class_trained = True

    def train_two_class(self, good_examples, bad_examples):
        assert self.type_ == "initiation"
        assert not self.for_global_option
        assert not self.two_class_trained

        xs = good_examples + bad_examples
        ys = [1 for _ in good_examples] + [0 for _ in bad_examples]

        self.two_class_svm.fit(xs, ys)
        self.two_class_trained = True
