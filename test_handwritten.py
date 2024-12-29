import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from PIL import Image

# Load the saved model parameters
weights_input_hidden = np.load('weights_input_hidden.npy')
weights_hidden_output = np.load('weights_hidden_output.npy')
bias_hidden = np.load('bias_hidden.npy')
bias_output = np.load('bias_output.npy')

# Load and process your handwritten image (e.g., a 28x28 image)
image_path = 'D:\\COMPLETEML\\python\\images\\Untitled.png'  # replace with your image path
img = Image.open(image_path).convert('L')  # Convert to grayscale
img = img.resize((28, 28))  # Resize to 28x28 (same as MNIST images)
img_array = np.array(img) / 255.0  # Normalize to [0, 1]
img_array = img_array.flatten().reshape(1, -1)  # Flatten the image

# Forward pass to make a prediction
hidden_layer_input = np.dot(img_array, weights_input_hidden) + bias_hidden
hidden_layer_output = np.maximum(0, hidden_layer_input)  # ReLU activation

output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
predictions = np.exp(output_layer_input - np.max(output_layer_input))  # Softmax
predictions = predictions / np.sum(predictions, axis=1, keepdims=True)

# Get the predicted class (digit)
predicted_class = np.argmax(predictions, axis=1)

# Display the image and the prediction
plt.imshow(img, cmap='gray')
plt.title(f'Predicted Class: {predicted_class[0]}')
plt.show()

print(f"Predicted Class: {predicted_class[0]}")
