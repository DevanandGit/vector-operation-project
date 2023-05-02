import numpy as np

# Define the matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Compute the QR decomposition
Q, R = np.linalg.qr(A)

# Print the results
print("Q =\n", Q)
print("R =\n", R)
