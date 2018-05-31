import numpy as np
from sklearn.utils import shuffle


np.random.seed(9999)

X0 = np.random.randn(100, 2) - 1
X1 = np.random.randn(100, 2) + 1
X = np.vstack([X0, X1])
t = np.vstack([np.zeros([100, 1]), np.ones([100, 1])])

X, t = shuffle(X, t)

X_train, X_test = X[:150], X[:50]
t_train, t_test = t[:150], t[:50]

# Model
W = np.random.randn(2, 1) * 0.01


def sigm(x):
    return 1/(1+np.exp(-x))


def NLL(y, t):
    return -np.mean(t*np.log(y) + (1-t)*np.log(1-y))


alpha = 1

# Training
for it in range(15):
    # Forward
    z = X_train @ W
    y = sigm(z)
    loss = NLL(y, t_train)

    # Loss
    print(f'Loss: {loss:.3f}')

    m = y.shape[0]

    dy = (y-t_train)/(m * (y - y*y))
    dz = sigm(z)*(1-sigm(z))
    dW = X_train.T @ (dz * dy)

    g = (t_train-y)/(y - y*y) * dz
    S = np.cov(g.T)
    A = np.cov(X_train.T)

    # Step
    W = W - alpha * (np.linalg.inv(A) @ dW * 1/S)

    # # Remark: naively the NG step is given by:
    # F = np.kron(np.cov(g.T), np.cov(X_train.T))
    # W = W - alpha * np.linalg.inv(F) @ dW

y = sigm(X_test @ W).ravel()
acc = np.mean((y >= 0.5) == t_test.ravel())

print(f'Accuracy: {acc:.3f}')
