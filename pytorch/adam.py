import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
import input_data
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('error')


np.random.seed(9999)
torch.manual_seed(9999)

mnist = input_data.read_data_sets('../MNIST_data', one_hot=False)

X_test = mnist.test.images
t_test = mnist.test.labels


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(784, 200, bias=False)
        self.fc2 = nn.Linear(200, 100, bias=False)
        self.fc3 = nn.Linear(100, 10, bias=False)

    def forward(self, x):
        a1 = self.fc1(x)
        h1 = F.relu(a1)
        a2 = self.fc2(h1)
        h2 = F.relu(a2)
        z = self.fc3(h2)

        return z


m = 128  # mb size
alpha = 0.001

# Model
model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=alpha)

# Visualization stuffs
losses = []

# Training
for i in range(1, 5000):
    X_mb, t_mb = mnist.train.next_batch(m)
    X_mb, t_mb = torch.from_numpy(X_mb), torch.from_numpy(t_mb).long()

    # Forward
    z = model.forward(X_mb)

    # Loss
    loss = F.cross_entropy(z, t_mb)
    loss.backward()

    if (i-1) % 100 == 0:
        print(f'Iter-{i}; Loss: {loss:.3f}')

    losses.append(loss if i == 1 else 0.99*losses[-1] + 0.01*loss)

    optimizer.step()

    # PyTorch stuffs
    optimizer.zero_grad()


z = model.forward(torch.from_numpy(X_test))
y = z.argmax(dim=1)
acc = np.mean(y.numpy() == t_test)

print(f'Accuracy: {acc:.3f}')
np.save('adam_losses.npy', losses)

