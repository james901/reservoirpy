import numpy as np

from scipy import linalg

from ._base import Readout


class PInvReadout(Readout):

    def __init__(self,
                 shape,
                 activation=None,
                 fb_activation=None,
                 fb_initializer=None,
                 Wout=None,
                 Wfb=Wfb):
        self.out_dim = out_dim
        self.Wout = Wout

    def __call__(self, states):
        ...

    def fit(self, states, teachers):
        self.Wout = teachers.dot(linalg.pinv(states))