from abc import ABC


class Readout(ABC):

    def __call__(self, states):
        raise NotImplementedError

    def fit(self, states, teachers):
        raise NotImplementedError

    def feedback(self,
                 state=None,
                 teacher=None):
        raise NotImplementedError
