# generate plots (traceplots, posterior plots)
import math
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, median, variance

def mode(data, step = 0.05, width = 0.05):
    start = min(data)
    end = max(data)
    max_count = 0
    max_i = -1
    for i in np.arange(start+width, end-width, step):
        count = 0
        for point in data:
            if i - width < point < i + width:
                count = count + 1
        if count > max_count:
            max_i = i
            max_count = count
    return max_i
