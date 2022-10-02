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

model_onehot = np.linalg.lstsq(data, onehot(target_vector))[0]
logits_onehot = np.matmul(data, model_onehot)
pred_onehot = logits_onehot.argmax(axis=1)
accuracy_onehot = (pred_onehot == target_vector).sum() / len(pred_onehot)

print()
print("accuracy_onehot is: ", accuracy_onehot) # accuracy WITH one-hot is 88%
print()

model_regular = np.linalg.lstsq(data, target_vector)[0]
pred_regular = np.matmul(data, model_regular).round()
accuracy_regular = (pred_regular == target_vector).sum() / len(pred_regular)
print()
print("accuracy_regular is: ", accuracy_regular) # accuracy WITHOUT one-hot is a measly 25%
print()

# Now see how well the non-onehot model does if we count it being almost right (i.e. getting the second most likely digit) as a success too. This is a bit flawed because to ascertain second most likely we're appealing to the onehot linear regression model, which is quite imperfect, but let's see if this says anything interesting.
backup_logits_onehot = logits_onehot.copy()
backup_logits_onehot[np.arange(len(pred_onehot)), pred_onehot] = 0
backup_pred_onehot = backup_logits_onehot.argmax(axis=1)
accuracy_regular_backup = (pred_regular == backup_pred_onehot).sum() / len(pred_onehot)
print("accuracy_regular when including second guess is: ", accuracy_regular + accuracy_regular_backup) # This gets it to 34% accuracy, so barely better. It still sucks. This is not a foolproof experiment since we're not comparing to any ground truth, but nonetheless surprising that it's still so bad.