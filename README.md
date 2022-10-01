# Linear Regression on MNIST

Linear regression is pretty effective on MNIST.

With one-hot encoding of the targets, it achieves 88% accuracy. Without one-hot encoding it gets a measly 25%.

(One-hot encoding means you use `[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0` as your label for an image instead of using `2`).

Run `mnist-linear-regression.py` to see without one-hot, and `mnist-linear-regression-one-hot.py` to see with.