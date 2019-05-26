# Natural Gradients
Collection of algorithms for approximating Fisher Information Matrix for Natural Gradient (also second order method in general). The problems used are the simplest possible (e.g. toy classification problem and MNIST), to highlight more on the algorithms themshelves.

## Remark (IMPORTANT)
The codes in this repo are using the empirical, and not the true Fisher matrix. So, take these with a grain of salt.
To actually compute the true Fisher matrix, we can do MC integration to approximate the expectation w.r.t. p(y | x, \theta): Sample y ~ p(y | x, \theta) repeatedly, and average the resulting outer products of the gradient.

### Dependencies

1. Python 3.5+
2. Numpy
3. Matplotlib

