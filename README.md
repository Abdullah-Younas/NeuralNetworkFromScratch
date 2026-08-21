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
- [ ] Milestone 6: Train Neural Network On Real Dataset
- [ ] Milestone 7: MNIST Digit Recognition
- [ ] Milestone 8: CNNs
- [ ] Milestone 9: Cats vs Dogs
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

## Milestone Train Neural Network On Real Dataset

Milestone 6A: Dataset

Create a text file:

height,weight,age,target
1.80,75,20,1
1.65,55,18,1
0.40,4,1,0
2.90,10,15,0

Then read it in C++.

Milestone 6B: Train On Multiple Samples

Currently:

x1 = 1.69;
x2 = 10;
x3 = 15;

Hardcoded.

Instead:

for each person in dataset
{
forward pass
calculate loss
backprop
update weights
}

This is the first time your network learns from data instead of a single example.

Milestone 6C: Save Weights

Example file:
w11=0.43
w12=0.88
w13=0.52
w21=0.35
...
After training:

ofstream file("weights.txt");

Write all weights.

Milestone 6D: Load Weights

At startup:
ifstream file("weights.txt");
Read weights.

If file doesn't exist:
use default/random weights

Status:
- [x] Complete

---