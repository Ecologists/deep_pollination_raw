'''Examples of organizer-provided metrics.
You can just replace this code by your own.
Make sure to indicate the name of the function that you chose as metric function
in the file metric.txt. E.g. mse_metric, because this file may contain more 
than one function, hence you must specify the name of the function that is your metric.'''

import numpy as np
import scipy as sp

def accuracy(solution, prediction):
    error = 0
    for sol, pred in zip(solution, prediction):
        if sol != pred:
            error += 1
    return 1 - (error / len(solution))

