import random
from copy import deepcopy
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC


class Classifier:
    def __init__(self, type_, for_global_option=False, for_goal_option=False, env_termination_checker=None):
        assert type_ in ["initiation", "termination"], "type_ can be 'initiation' or 'termination'"
        assert not (for_global_option and for_goal_option), "it cant be both global and goal"

        self.type_ = type_
        self.for_global_option = for_global_option
        self.for_goal_option = for_goal_option

        self.for_last_option = False
        self.env_termination_checker = env_termination_checker
        self.one_class_svm = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")

        self.one_class_trained = False
        self.one_class_refined = False
        self.good_examples_to_sample = []
        self.initial_state = None

        if self.type_ == "termination" and (self.for_global_option or self.for_goal_option):
            assert self.env_termination_checker is not None, \
                "if goal or global option, termination checker should be provided"
        elif self.type_ == "termination":
            assert self.env_termination_checker is None, "why provide termination checker for non global or goal"

    def check(self, x) -> bool:
        if self.type_ == "initiation" and self.for_global_option:
            return True

        if self.type_ == "termination" and (self.for_global_option or self.for_goal_option):
            return self.env_termination_checker(x)

        if self.for_last_option and x.tolist() == self.initial_state.tolist():
            return True

        return self.one_class_svm.predict([x])[0] == 1

    def sample(self):
        assert self.type_ == "termination", "sampling only valid for termination classifiers"
        assert self.one_class_trained, "at least one_class must be trained"

        return random.sample(self.good_examples_to_sample, k=1)[0]

    def train_one_class(self, xs, initial_state):
        assert self.type_ == "initiation", "only initiation classifiers can be trained"
        assert not self.for_global_option, "global option classifiers cannot be trained"
        assert not self.one_class_trained, "one_class shouldn't be trained yet to train"

        self.good_examples_to_sample = deepcopy(xs)

        for arr in xs:
            if arr.tolist() == initial_state.tolist():
                self.for_last_option = True
                self.initial_state = initial_state

        self.one_class_svm.fit(xs)
        self.one_class_trained = True

    def train_two_class(self, good_examples, bad_examples):
        assert self.type_ == "initiation", "only initiation classifiers can be trained"
        assert not self.for_global_option, "global option classifiers cannot be trained"
        assert not self.one_class_refined, "one_class shouldn't be re-trained"

        # we are also using the data to train one class classifier for good_examples
        good_examples = good_examples + deepcopy(self.good_examples_to_sample)
        self.good_examples_to_sample = deepcopy(good_examples)

        xs = np.array(good_examples + bad_examples)
        ys = np.array([1 for _ in good_examples] + [0 for _ in bad_examples])

        two_class_svm = SVC(kernel="rbf", gamma="scale", class_weight="balanced")
        two_class_svm.fit(xs, ys)

        training_predictions = two_class_svm.predict(xs)
        positive_training_examples = xs[training_predictions == 1]

        self.one_class_svm.fit(positive_training_examples)
        self.one_class_refined = True
