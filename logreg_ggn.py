"""
Second order method with Gauss-Newton Matrix (GGN) as the curvature matrix.
"""
import numpy as np
from sklearn.utils import shuffle


np.random.seed(9999)

X0 = np.random.randn(100, 2) - 1
X1 = np.random.randn(100, 2) + 1
X = np.vstack([X0, X1])
y = np.vstack([np.zeros([100, 1]), np.ones([100, 1])])

X, y = shuffle(X, y)

X_train, X_test = X[:150], X[:50]
y_train, y_test = y[:150], y[:50]

# Model
W = np.random.randn(2, 1) * 0.01


def sigm(x):
    """ Sigmoid function """
    return 1/(1+np.exp(-x))


def L(y, z):
    """ Loss function; y: target, z: model's output (pred) """
    return -np.mean(y*np.log(z) + (1-y)*np.log(1-z))


def grads(y, z):
    """ Return all gradients/jacobians in the computation graph """
    dLdz = z - y
    dLdW = (X_train * dLdz)[:, :, None]  # m x 2 x 1

    return dLdz, dLdW


alpha = 1

# Training
for it in range(15):
    m = y_train.shape[0]

    # Forward
    o = X_train @ W
    z = sigm(o)
    loss = L(y_train, z)

    # Loss
    print(f'Loss: {loss:.3f}')

    # Backward
    dLdz, dLdW = grads(y_train, z)

    # Hessian of loss wrt. output
    H_L = np.ones([1, 1])

    # Gauss-Newton matrix
    G = np.mean(dLdW @ H_L @ dLdW.transpose((0, 2, 1)), axis=0)

    # Scale gradients
    dLdW = np.mean(dLdW, axis=0)

    # # Step
    W = W - alpha * np.linalg.inv(G) @ dLdW

z = sigm(X_test @ W).ravel()
acc = np.mean((z >= 0.5) == y_test.ravel())
print(f'Accuracy: {acc:.3f}')
