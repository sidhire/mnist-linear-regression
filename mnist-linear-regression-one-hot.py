import numpy as np
import pickle

# Took the MNIST data (10000 images) from PyTorch and pickled it
MNIST_DATA_FILENAME = "mnist-data.pickle"
NUM_CLASSES = 10

with open(MNIST_DATA_FILENAME, "rb") as f:
    obj = pickle.load(f)

data = obj["data"] # shape is (10000, 784)
target_vector = obj["target"] # shape is (10000,)

def onehot(index_vector):
    return np.identity(NUM_CLASSES)[index_vector]

target = onehot(target_vector) # shape is (10000, 10)

# Least squares regression
# Computes the vector x that approximately solves the equation a @ x = b
regression = np.linalg.lstsq(data, target)
model = regression[0] # this is the vector x

pred = np.matmul(data, model).argmax(axis=1)

accuracy = (pred == target_vector).sum() / len(pred)
print("accuracy is: ", accuracy) # accuracy WITH one-hot is 88%

# It's basically same if we hold out 10% of the data for testing.