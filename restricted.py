import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import binarize

class RBM:
    def __init__(self, n_visible, n_hidden, learning_rate=0.1):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.weights = np.random.normal(0, 0.1, (n_visible, n_hidden))
        self.v_bias = np.zeros(n_visible)
        self.h_bias = np.zeros(n_hidden)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, data, epochs=10, batch_size=10):
        for epoch in range(epochs):
            np.random.shuffle(data)
            for batch_start in range(0, len(data), batch_size):
                batch = data[batch_start:batch_start + batch_size]

                # Positive phase
                pos_hidden_probs = self.sigmoid(batch @ self.weights + self.h_bias)
                pos_hidden_states = (pos_hidden_probs > np.random.rand(len(batch), self.n_hidden)).astype(float)
                pos_associations = batch.T @ pos_hidden_probs

                # Negative phase
                neg_visible_probs = self.sigmoid(pos_hidden_states @ self.weights.T + self.v_bias)
                neg_hidden_probs = self.sigmoid(neg_visible_probs @ self.weights + self.h_bias)
                neg_associations = neg_visible_probs.T @ neg_hidden_probs

                # Update parameters
                self.weights += self.lr * (pos_associations - neg_associations) / len(batch)
                self.v_bias += self.lr * np.mean(batch - neg_visible_probs, axis=0)
                self.h_bias += self.lr * np.mean(pos_hidden_probs - neg_hidden_probs, axis=0)

    def reconstruct(self, data):
        hidden_probs = self.sigmoid(data @ self.weights + self.h_bias)
        hidden_states = (hidden_probs > np.random.rand(len(data), self.n_hidden)).astype(float)
        reconstructed = self.sigmoid(hidden_states @ self.weights.T + self.v_bias)
        return reconstructed

# Load and preprocess MNIST data
digits = load_digits()
X = digits.data / 16.0  # Normalize to [0,1]
X = binarize(X, threshold=0.5)  # Binarize the data

# Initialize and train RBM
n_visible = X.shape[1]  # 64 for 8x8 MNIST digits
n_hidden = 100
rbm = RBM(n_visible, n_hidden)
rbm.train(X, epochs=20, batch_size=10)

# Test reconstruction
sample = X[:5]
reconstructed = rbm.reconstruct(sample)
print("Original sample shape:", sample.shape)
print("Reconstructed shape:", reconstructed.shape)
