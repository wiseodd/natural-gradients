"""
João F. Henriques, Sebastien Ehrhardt, Samuel Albanie, Andrea Vedaldi
"Small steps and giant leaps: Minimal Newton solvers for Deep Learning"
arXiv preprint, 2018
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


def L(y, p):
    """ Loss function; y: target, z: model's output (pred) """
    return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))


def grads(y, o):
    """ Return all gradients/jacobians in the computation graph """
    dLdo = o - y
    dLdW = (X_train * dLdo)[:, :, None]  # (m, 2, 1)

    return dLdo, dLdW


alpha = 1
lam = 1
beta = 0.1
rho = 0.9
z = np.zeros_like(W)  # (2, 1)

# Training
for it in range(5):
    m = y.shape[0]

    # Forward
    o = X_train @ W
    p = sigm(o)
    loss = L(y_train, p)

    # Loss
    print(f'Loss: {loss:.3f}')

    # Backward
    dLdo, dLdW = grads(y_train, o)

    # Gradient of loss wrt. output
    J_L = dLdo[:, :, None].mean(0)  # (m, 1, 1) -> (1, 1)

    # Hessian of loss wrt. output
    H_L = (o * (1-o))[:, :, None].mean(0)  # (m, 1, 1) -> (1, 1)

    # Scale gradients
    J_W = dLdW.mean(0)  # (m, 2, 1) -> (2, 1)

    # Step
    delta_z = J_W @ (H_L @ J_W.T @ z + J_L) + lam*z
    z = rho*z + beta*delta_z
    W = W + alpha*z

z = sigm(X_test @ W).ravel()
acc = np.mean((z >= 0.5) == y_test.ravel())
print(f'Accuracy: {acc:.3f}')