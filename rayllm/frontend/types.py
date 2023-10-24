from typing import List


class Selection:
    def __init__(self, button, choices: List):
        self.button = button
        self.choices = choices

    def to_list(self):
        return [self.button] + self.choices

    @classmethod
    def from_list(cls, lst):
        return lst[0], list(lst[1:])
