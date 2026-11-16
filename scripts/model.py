"""The model: a small ("shallow") neural network, written in plain numpy so every step is visible.

What this does, in plain words
------------------------------
Question the model answers: given the pristine and exposed pictures of one sensor, how long
did the sensor last?  The answer is log10(time to failure in ms).  We guess the LOG because
times run from about 10 seconds to over an hour, so "off by a factor of 2" is the natural
size of a mistake.

How it works (one hidden layer):
    inputs  ->  HIDDEN numbers  ->  one number
 1. Inputs: both pictures shrunk to PIXELS x PIXELS (2 x PIXELS x PIXELS numbers) plus four
    0/1 numbers saying which sensor it is (U1-U4).  Every input is rescaled to "typical = 0,
    spread = 1" using the TRAINING samples only.
 2. Hidden layer: each hidden number is a weighted sum of all inputs; negative sums are set
    to zero (that is the "ReLU" step).
 3. Output: one weighted sum of the hidden numbers = the guess.
 Training nudges the weights to shrink the average squared mistake.  `WEIGHT_DECAY` pulls
 weights toward zero so the network cannot simply memorise 300 samples.  "Adam" is a standard
 way of choosing how big each nudge is.

This is deliberately a WEAK baseline: the point of Part I is that the whole chain works.
To try your own model in Part II (a CNN, DINOv3, ...), rewrite this file and train.py.  The scoring
step (evaluate.py) only needs PIXELS, SEED, Baseline.load(path) and Baseline.predict(x, sensor);
see the README section "Swapping in your own model".
"""
import numpy as np

from split import SEED          # the same seed (5937) as the data split

# These settings came from trying a handful of values and watching the CHECK-SET loss (never the
# held-out evaluation file).  They are not tuned any further: tuning is Part II.
PIXELS = 32            # each picture is shrunk to PIXELS x PIXELS
HIDDEN = 16            # size of the hidden layer
EPOCHS = 60            # one epoch = one pass over all training samples
BATCH = 32             # samples per nudge ("step")
LEARNING_RATE = 3e-4   # size of each nudge
WEIGHT_DECAY = 0.1     # pull of the weights toward zero
VAL_FOLD = 0           # this cross-validation fold is the check set while training
N_SENSORS = 4


def make_inputs(x, sensor):
    """Flatten the pictures and add the 0/1 sensor label. x: [N, 2, P, P]; sensor: [N] of 0-3."""
    flat = x.reshape(len(x), -1).astype(np.float64)
    one_hot = np.eye(N_SENSORS)[sensor]
    return np.concatenate([flat, one_hot], axis=1)


def forward(p, X):
    """Run the network.  p = weights, X = inputs [N, n_in].  Returns guesses and what backward needs."""
    z = X @ p["W1"] + p["b1"]          # hidden sums
    h = np.maximum(z, 0.0)             # ReLU: negatives become zero
    return h @ p["W2"] + p["b2"], (z, h)


def loss_and_grads(p, X, y, weight_decay):
    """Average squared mistake, and how it changes with every weight (backpropagation)."""
    out, (z, h) = forward(p, X)
    err = out - y
    loss = float(np.mean(err ** 2))
    d_out = 2.0 * err / len(y)                         # d loss / d guess
    dz = np.outer(d_out, p["W2"]) * (z > 0)            # pushed back through the ReLU
    grads = {
        "W2": h.T @ d_out + weight_decay * p["W2"],
        "b2": np.array([d_out.sum()]),
        "W1": X.T @ dz + weight_decay * p["W1"],
        "b1": dz.sum(axis=0),
    }
    return loss, grads


class Adam:
    """Adam: gives every weight its own step size from the recent size of its gradient."""

    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, beta1, beta2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class Baseline:
    """The model plus the rescaling it needs.  train.py trains it; evaluate.py asks it for guesses."""

    def __init__(self):
        self.params = None
        self.x_mean = self.x_std = self.y_mean = self.y_std = None

    def fit_scalers(self, x, sensor, y_log):
        """Remember the training samples' typical values, so inputs and answers can be rescaled."""
        X = make_inputs(x, sensor)
        self.x_mean = X.mean(axis=0)
        self.x_std = np.maximum(X.std(axis=0), 1e-2)
        self.x_mean[-N_SENSORS:], self.x_std[-N_SENSORS:] = 0.0, 1.0      # leave the 0/1 sensor label alone
        self.y_mean, self.y_std = float(y_log.mean()), float(y_log.std())

    def init_weights(self, n_in, rng):
        self.params = {
            "W1": rng.normal(0.0, np.sqrt(2.0 / n_in), (n_in, HIDDEN)),
            "b1": np.zeros(HIDDEN),
            "W2": rng.normal(0.0, np.sqrt(1.0 / HIDDEN), HIDDEN),
            "b2": np.zeros(1),
        }

    def scaled_inputs(self, x, sensor):
        return (make_inputs(x, sensor) - self.x_mean) / self.x_std

    def scaled_answers(self, y_log):
        return (y_log - self.y_mean) / self.y_std

    def predict(self, x, sensor):
        """Guess log10(time to failure in ms) for each sample."""
        out, _ = forward(self.params, self.scaled_inputs(x, sensor))
        return out * self.y_std + self.y_mean

    def n_parameters(self):
        return int(sum(v.size for v in self.params.values()))

    def save(self, path):
        with open(path, "wb") as fh:
            np.savez(fh, x_mean=self.x_mean, x_std=self.x_std, y_mean=self.y_mean, y_std=self.y_std,
                     **{f"param_{k}": v for k, v in self.params.items()})

    @classmethod
    def load(cls, path):
        model = cls()
        with np.load(path) as f:
            model.x_mean, model.x_std = f["x_mean"], f["x_std"]
            model.y_mean, model.y_std = float(f["y_mean"]), float(f["y_std"])
            model.params = {k[len("param_"):]: f[k] for k in f.files if k.startswith("param_")}
        return model
