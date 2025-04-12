import os
import numpy as np

def save(filename, data):
    """
    Save data to a file.
    """
    folder = os.path.dirname(filename)
    if folder:
        if not os.path.exists(folder):
            os.makedirs(folder)
    np.savetxt(filename, data, delimiter=",")