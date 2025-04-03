import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
data = mnist['data']
target = mnist['target'].astype(int)


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(data_scaled, target, test_size=0.2, random_state=42)
from sklearn.neural_network import BernoulliRBM

# Train individual RBMs layer by layer
rbm1 = BernoulliRBM(n_components=128, learning_rate=0.1, n_iter=10, random_state=42)
X_train_rbm1 = rbm1.fit_transform(X_train)

rbm2 = BernoulliRBM(n_components=64, learning_rate=0.1, n_iter=10, random_state=42)
X_train_rbm2 = rbm2.fit_transform(X_train_rbm1)

# Apply the same transformations on the test set
X_test_rbm1 = rbm1.transform(X_test)
X_test_rbm2 = rbm2.transform(X_test_rbm1)

# Check the shape of the transformed data at each layer
print(f"Shape of data after first RBM: {X_train_rbm1.shape}")
print(f"Shape of data after second RBM: {X_train_rbm2.shape}")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train a logistic regression classifier on the extracted features
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_rbm2, y_train)

# Predict on the test data
y_pred = classifier.predict(X_test_rbm2)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification accuracy after stacking RBMs: {accuracy * 100:.2f}%")
