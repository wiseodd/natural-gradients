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
W = np.random.randn(784, 10) * 0.01


def softmax(x):
    ex = np.exp(x - np.max(x, axis=1)[:, None])
    return ex / ex.sum(axis=1)[:, None]


def NLL(z, t):
    return -np.mean(np.sum(t*np.log(softmax(z)), axis=1))


m = 64  # mb size
alpha = 0.001

# Visualization stuffs
losses = []

# Training
for i in range(1, 5000):
    X_mb, t_mb = mnist.train.next_batch(m)
    t_mb_idx = t_mb.argmax(axis=1)

    # Forward
    z = X_mb @ W
    loss = NLL(z, t_mb)

    # Loss
    if (i-1) % 100 == 0:
        print(f'Iter-{i}; Loss: {loss:.3f}')

    losses.append(loss if i == 1 else 0.99*losses[-1] + 0.01*loss)

    m = z.shape[0]

    # Gradients
    dz = softmax(z)
    dz[range(m), t_mb_idx] -= 1
    dz /= m
    g = X_mb.T @ dz

    # Step
    W = W - alpha * g

y = softmax(X_test @ W).argmax(axis=1)
acc = np.mean(y == t_test.argmax(axis=1))

print(f'Accuracy: {acc:.3f}')
np.save('sgd_losses.npy', losses)

