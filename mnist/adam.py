import numpy as np
import input_data
from sklearn.utils import shuffle


np.random.seed(9999)

mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)

X_train = mnist.train.images
t_train = mnist.train.labels

X_test = mnist.test.images
t_test = mnist.test.labels

X_train, t_train = shuffle(X_train, t_train)

# Model
W1 = np.random.randn(784, 100) * 0.01
W2 = np.random.randn(100, 10) * 0.01


def softmax(x):
    ex = np.exp(x - np.max(x, axis=1)[:, None])
    return ex / ex.sum(axis=1)[:, None]


def NLL(z, t):
    return -np.mean(np.sum(t*np.log(softmax(z) + eps), axis=1))


m = 200  # mb size
alpha = 0.001
rho1 = 0.9  # Decay for F
rho2 = 0.999  # Momentum
s1 = np.zeros_like(W1)
r1 = np.zeros_like(W1)
s2 = np.zeros_like(W2)
r2 = np.zeros_like(W2)
eps = 1e-8

# Visualization stuffs
losses = []

# Training
for i in range(1, 5000):
    X_mb, t_mb = mnist.train.next_batch(m)
    t_mb_idx = t_mb.argmax(axis=1)

    # Forward
    a = X_mb @ W1
    h = np.maximum(a, 0)
    z = h @ W2
    loss = NLL(z, t_mb)

    # Loss
    if (i-1) % 100 == 0:
        print(f'Iter-{i}; Loss: {loss:.3f}')

    losses.append(loss if i == 1 else 0.99*losses[-1] + 0.01*loss)

    m = z.shape[0]

    # Gradients
    dz = softmax(z)
    dz[range(dz.shape[0]), t_mb_idx] -= 1  # m*10
    dz /= m
    dW2 = h.T @ dz  # 100*10
    dh = dz @ W2.T  # m*100
    dh[a < 0] = 0  # ReLU
    dW1 = X_mb.T @ dh  # 784*100

    # Moments
    s1 = rho1*s1 + (1-rho1)*dW1
    r1 = rho2*r1 + (1-rho2)*(dW1*dW1)
    s2 = rho1*s2 + (1-rho1)*dW2
    r2 = rho2*r2 + (1-rho2)*(dW2*dW2)
    #  r = rho2*r + (1-rho2)*(m*g*g)  # Corresponds to diagonal approx. of FIM

    # Bias correction
    s1_ = s1/(1-rho1**i)
    r1_ = r1/(1-rho2**i)
    s2_ = s2/(1-rho1**i)
    r2_ = r2/(1-rho2**i)

    # Step
    delta1 = s1_ / (np.sqrt(r1_) + eps)
    delta2 = s2_ / (np.sqrt(r2_) + eps)
    #  delta = s_ / (r_ + eps)  # Inverse of diagonal FIM
    #  W = W - alpha * g  # SGD update
    W1 = W1 - alpha * delta1
    W2 = W2 - alpha * delta2

y = softmax(np.maximum(X_test @ W1, 0) @ W2).argmax(axis=1)
acc = np.mean(y == t_test.argmax(axis=1))

print(f'Accuracy: {acc:.3f}')
np.save('adam_losses.npy', losses)

