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





x = np.arange(100).reshape(10, 10)

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def go_fast(a): # Function is compiled to machine code when called the first time
    trace = 0.0
    for i in range(a.shape[0]):   # Numba likes loops
        trace += np.tanh(a[i, i]) # Numba likes NumPy functions
    return a + trace              # Numba likes NumPy broadcasting

print(go_fast(x))