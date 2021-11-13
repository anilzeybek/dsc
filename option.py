class Option:
    def __init__(self, budget: int) -> None:
        self.budget = budget

    def check_in_initiation_set(self, obs) -> bool:
        return True
