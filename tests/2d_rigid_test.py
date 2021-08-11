#!/usr/bin/python
import numpy as np

# Input points
from_pt = [[1,1], [1,2], [2,2], [2,1]] # A rectangle at 1,1
to_pt =   [[2,2], [4,4], [6,2], [4,0]] # The same, transformed

# Fill the matrices
A_data = []
for pt in from_pt:
  A_data.append( [-pt[1], pt[0], 1, 0] )
  A_data.append( [ pt[0], pt[1], 0, 1] )

b_data = []
for pt in to_pt:
  b_data.append(pt[0])
  b_data.append(pt[1])

# Solve
A = np.matrix( A_data )
b = np.matrix( b_data ).T
c = np.linalg.lstsq(A, b, rcond=None)[0].T
c = np.array(c)[0]

print("Solved coefficients:")
print(c)

print("Translated 'from_pt':")
# These will be identical to 'to_pt' since
# our example transformation is exact
for pt in from_pt:
  print ("%f, %f" % (
    c[1]*pt[0] - c[0]*pt[1] + c[2],
    c[1]*pt[1] + c[0]*pt[0] + c[3] ))