import numpy as np
import pickle

# Took the MNIST data (10000 images) from PyTorch and pickled it
MNIST_DATA_FILENAME = "mnist-data.pickle"

with open(MNIST_DATA_FILENAME, "rb") as f:
    obj = pickle.load(f)

data = obj["data"] # shape is (10000, 784)
target = obj["target"] # shape is (10000,)

# Least squares regression
# Computes the vector x that approximately solves the equation a @ x = b
regression = np.linalg.lstsq(data, target)
model = regression[0] # this is the vector x

# Verified that this rounding method is identical even though it has that weird behavior
# It doesn't end up mattering because every output has more than 1 decimal place
pred = np.matmul(data, model).round()

accuracy = (pred == target).sum() / len(pred)
print("accuracy is: ", accuracy) # accuracy WITHOUT one-hot is a measly 25%

# See mnist-linear-regression-one-hot.py for a version that gets 88%.
