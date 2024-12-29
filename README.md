# Digit Recognition Neural Network

## Overview
This project implements a **Feedforward Neural Network (FNN)**, which is a type of **Artificial Neural Network (ANN)** for recognizing handwritten digits (0-9) using the MNIST dataset. The network achieves **97% accuracy** by learning the patterns in the data through forward propagation, backpropagation, and gradient descent. The goal of this project is to demonstrate the basic concepts of neural networks and how they can be trained to make predictions based on labeled data.

## Concepts Covered
This neural network is built around the foundational concepts of machine learning and neural networks:
- **Neural Network Architecture:**
  - **Input Layer:** 784 neurons, one for each pixel in the 28x28 grayscale image.
  - **Hidden Layer(s):** Layers that learn patterns and features from the input data.
  - **Output Layer:** 10 neurons, each representing a digit (0-9) to classify the input image.
- **Forward Propagation:** The process where input data is passed through the network, generating predictions.
- **Backpropagation:** A method for calculating gradients and updating weights to minimize the error using the chain rule.
- **Gradient Descent:** The optimization technique that adjusts weights to minimize the error (loss function) and improve the model’s performance.

## Key Features
- **MNIST Dataset:** The dataset contains 60,000 training images and 10,000 test images of handwritten digits. Each image is 28x28 pixels, and each label represents a digit between 0 and 9.
- **Gradient Descent:** The network uses gradient descent to update the weights of the model, adjusting them to minimize the difference between predicted and actual values.
- **Accuracy:** After training, the network achieves **97% accuracy** on the test dataset, demonstrating its ability to classify unseen images correctly.

## How It Works
1. **Forward Pass:**
   - The input image (28x28 pixels) is flattened into a 784-dimensional vector.
   - The input is passed through the hidden layers using a set of weights and biases.
   - Each layer uses an activation function (such as sigmoid or ReLU) to transform the data.
   - The final output layer computes the probabilities for each class (digit 0-9).

2. **Loss Calculation:**
   - The error (loss) is calculated using **cross-entropy loss** to compare the network’s predictions with the true labels.

3. **Backpropagation:**
   - Gradients are computed using the chain rule, which tells us how much each weight contributed to the error.
   - This gradient information is used to update the weights during the next step.

4. **Gradient Descent:**
   - The model’s weights are updated using gradient descent, adjusting them in the direction that minimizes the loss.
   - The learning rate controls how big each weight update is.

5. **Model Training:**
   - The process of forward propagation, loss calculation, backpropagation, and weight updates is repeated for multiple iterations (epochs) until the model achieves the desired accuracy.

## Technologies Used
- **Python:** Programming language used to build the neural network.
- **NumPy:** Library for numerical operations, such as matrix multiplication and gradient calculations.
- **Matplotlib:** Library used to visualize the dataset and the model's performance.

## Installation
To run this project locally, ensure you have the following dependencies:

```bash
pip install numpy matplotlib
```

## Usage
Clone the repository:

```bash
git clone https://github.com/sivajeetsabdakar/digit-recognition-neural-network.git
cd digit-recognition-neural-network
```
Run the neural_network.py script to train the model and test its accuracy:

```
python neural_network.py
```

The model will train on the MNIST dataset, and you'll see the accuracy printed out after each epoch.

# Results
The neural network achieves a 97% accuracy on the MNIST test set, making it capable of recognizing handwritten digits with high precision.

# Next Steps
- Hyperparameter Tuning: Experiment with different architectures, learning rates, and activation functions to improve performance.
- Deep Learning: Extend this basic model to deeper networks with more hidden layers.
- Deployment: Explore options to deploy the model for real-time digit recognition using frameworks like Flask or TensorFlow.

# Conclusion
This project helped solidify my understanding of the core principles of neural networks. By building this simple model, I learned how data flows through a network, how backpropagation works, and how gradient descent is used to optimize the network’s weights. With 97% accuracy, this model demonstrates the potential of neural networks in image classification tasks.

# Acknowledgments
3Blue1Brown: The explanation of neural networks from the 3Blue1Brown YouTube channel was a huge inspiration and helped me understand the mathematics behind these algorithms.
