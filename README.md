# Neural Network From Scratch (C++)

## Goal

Learn how neural networks, computer vision, and language models work from first principles without using high level AI frameworks.

---

## Progress

- [x] Milestone 1: Single Perceptron
- [x] Milestone 2: Tiny Classifier
- [x] Milestone 3: Multiple Neurons
- [x] Milestone 4: Teach The Network
- [x] Milestone 5: Backpropagation
- [x] Milestone 6: Train Neural Network On Real Dataset
- [x] Milestone 7: MNIST Digit Recognition
- [x] Milestone 8: CNNs
- [x] Milestone 9: Cats vs Dogs
- [ ] Milestone 10: Language Models
- [ ] Milestone 11: Transformers

---

## Roadmap

### Milestone 1: Single Perceptron ✅

Build a perceptron that makes a simple decision.

Project:
- Movie decision classifier

Concepts:
- Inputs
- Weights
- Bias
- Step Activation Function

Status:
- [x] Complete

---

### Milestone 2: Tiny Classifier ✅

Build a simple classifier using numerical inputs.

Example:
- Height
- Weight

Output:
- Human
- Not Human

Research:
- Activation Functions
- Sigmoid
- ReLU

Status:
- [x] Complete

---

### Milestone 3: Multiple Neurons

Build a small feed-forward neural network.

Architecture:

Inputs
↓
Hidden Layer
↓
Output

Research:
- Dense Layers
- Feed Forward Networks

Concepts:
- Multiple Neurons
- Hidden Layers

Built:

- 3 input features
- 1 hidden layer with 3 neurons
- Sigmoid activation
- 1 output neuron

Learned:

- Information flows from inputs to hidden neurons.
- Hidden neurons produce intermediate outputs.
- The output neuron combines hidden neuron outputs.
- Manually chosen weights and biases do not produce meaningful predictions.
- Neural networks require a way to measure error and update weights.

Status:
- [x] Complete

---

### Milestone 4 - Loss Functions

because now you have a network that can actually make mistakes.

Allow the network to measure its mistakes.

Research:
- Loss Functions
- Mean Squared Error (MSE)
- Cross Entropy

Status:
- [x] Complete

---

## Milestone 5 - Backpropagation

Implemented backpropagation from scratch in C++.

Features:
- Feed forward neural network
- Sigmoid activation
- Mean Squared Error (MSE)
- Error propagation through hidden layers
- Weight updates using gradient descent

Results:

Epoch 0
Prediction: 0.619406
MSE: 0.383664

Epoch 1000
Prediction: 0.0501419
MSE: 0.00251421

Observation:
The network successfully reduced prediction error over multiple epochs, demonstrating learning through weight updates.

Status:
- [x] Complete

## Milestone 6 - Train Neural Network On Real Dataset

Finally Complete my Human Classifies

Features:
- Trains on real Dataset of humans and non humans
- Loads weights, improves them and saves them into a file
- Load biases, improves them and saves them into a file
- Uses a 3 9 1 layer system with a total of 36 weights in total
- Has 2 modes Training Mode and Predicting Mode

Observation:
The network trained this way performs with extreme probability and produces almost perfect results.

Status:
- [x] Complete

## Milestone 7 Completed - MNIST Neural Network C++

Built and trained a neural network from scratch in C++ on MNIST.

Features:
- 60,000 training images & 10,000 test images
- 784 → 128 → 10 architecture
- 97.1% test accuracy

Observation: It takes ~1 hour for 10 epochs. 10,000 epochs would take ~ 42 DAYS 💀
God damn

Status:
- [x] Complete

## Milestone 8: Convolutional Neural Networks (CNNs)

Research:
- Convolution
- Pooling
- CNNs

Question:
- Why are CNNs better at images?

Concepts:
- Feature Extraction
- Spatial Information

Status:
- [x] Complete

### Milestone 9: Cats vs Dogs Classifier

Goal:
-   Image
      ↓
-    CNN
      ↓
- Cat or Dog

Concepts:
- Binary Classification
- Image Recognition

learned:
- Learned that CUDA is fricking impossible to understand and that in this case my Neural Network is not learning after adding cuda for some reason

Status:
- [x] Complete

---