"""Numerical equations and integration routines for the local EMS model."""
import numpy as np

class SavedSlices:

    def __init__(self, saved):
        self.saved = saved

    def sol(self, times):
        return np.stack([self.saved[round(float(time), 10)] for time in times], axis=1)
