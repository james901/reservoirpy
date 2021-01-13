from ._base import Readout


class PInvReadout(Readout):

    def __init__(self, out_dim, Wout=None):
        self.out_dim = out_dim
        self.Wout = Wout

    def __call__(self, states):
        ...
    
    def fit(self, states, teachers):
        self.Wout = teachers.dot(linalg.pinv(states))