from numba import jit
import numpy as np
import math

@jit
def hypot(x, y):
  return math.sqrt(x*x + y*y)

# Numba function
print(hypot(3.0, 4.0))

# Python function
print(hypot.py_func(3.0, 4.0))
