import numpy as np

class JointConfigTool(object):
    def __init__(self, bounds):
        assert bounds.shape[1] == 2
        self._min = bounds[:,0]
        self._max = bounds[:,1]

    def validate_bounds(self, js):
        return np.all(js>self._min and js<self._max)

    def normalize(self, js):
        js = 1. * (js - self._min) / (self._max - self._min)
        js = js * 2. - 1.
        return js

    def unnormalize(self, js):
        js = (js + 1.) / 2.
        js = 1. * js * (self._max - self._min) + self._min
        return js
    
    def clamp(self, js):
        min_ = np.ones(js.shape) * self._min
        max_ = np.ones(js.shape) * self._max
        js = np.minimum(np.maximum(js, min_), max_)
        return js
    