import numpy as np
from tensorflow.keras.datasets import mnist

# Load MNIST dataset from TensorFlow
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize input data to the range [0, 1]
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0  # Flatten images
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0  # Flatten images

# Initialize parameters
input_size = 784  # 28x28 pixels flattened
hidden_size = 64  # Number of neurons in the hidden layer
output_size = 10  # 10 classes (digits 0-9)

np.random.seed(42)
weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
bias_hidden = np.zeros((1, hidden_size))
bias_output = np.zeros((1, output_size))

# Activation function: ReLU
def relu(x):
    return np.maximum(0, x)

# Derivative of ReLU
def relu_derivative(x):
    return (x > 0).astype(float)

# Softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability fix
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Loss function: Categorical Cross-Entropy
def cross_entropy_loss(y_pred, y_true):
    m = y_pred.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    return np.sum(log_likelihood) / m

# Training loop
epochs = 10
learning_rate = 0.1
batch_size = 64

for epoch in range(epochs):
    for i in range(0, X_train.shape[0], batch_size):
        # Mini-batch
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        # Forward pass
        hidden_layer_input = np.dot(X_batch, weights_input_hidden) + bias_hidden
        hidden_layer_output = relu(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
        predictions = softmax(output_layer_input)

        # Compute loss
        loss = cross_entropy_loss(predictions, y_batch)

        # Backward pass
        # Gradients for output layer
        m = y_batch.shape[0]
        y_one_hot = np.zeros((m, output_size))
        y_one_hot[np.arange(m), y_batch] = 1

        d_output = predictions - y_one_hot
        d_weights_hidden_output = np.dot(hidden_layer_output.T, d_output) / m
        d_bias_output = np.sum(d_output, axis=0, keepdims=True) / m

        # Gradients for hidden layer
        d_hidden = np.dot(d_output, weights_hidden_output.T) * relu_derivative(hidden_layer_input)
        d_weights_input_hidden = np.dot(X_batch.T, d_hidden) / m
        d_bias_hidden = np.sum(d_hidden, axis=0, keepdims=True) / m

        # Parameter updates
        weights_input_hidden -= learning_rate * d_weights_input_hidden
        weights_hidden_output -= learning_rate * d_weights_hidden_output
        bias_hidden -= learning_rate * d_bias_hidden
        bias_output -= learning_rate * d_bias_output

    # Print loss for each epoch
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

# Evaluate on test data
hidden_layer_input_test = np.dot(X_test, weights_input_hidden) + bias_hidden
hidden_layer_output_test = relu(hidden_layer_input_test)
output_layer_input_test = np.dot(hidden_layer_output_test, weights_hidden_output) + bias_output
test_predictions = softmax(output_layer_input_test)

accuracy = np.mean(np.argmax(test_predictions, axis=1) == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model weights and biases after training
np.save('weights_input_hidden.npy', weights_input_hidden)
np.save('weights_hidden_output.npy', weights_hidden_output)
np.save('bias_hidden.npy', bias_hidden)
np.save('bias_output.npy', bias_output)

print("Model saved!")