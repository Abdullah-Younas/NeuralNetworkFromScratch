// Neural Network.cpp : This file contains the 'main' function. Program execution begins and ends there.
//Milestone 2: Tiny classifies that classifies whether the given height & weight is human or not.
//The solution is not correct as of now it doesn't classifies correctly whether the inputs are human or not

#include <iostream>
#include <math.h>
#include <numbers>

using namespace std;

double Sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

int main()
{
    //Inputs
    double x1; //Height in meter
    double x2; //Weight in kg

    //Weights
    double w1 = 2.0; // Height
    double w2 = 4.0; // Weight

    //Bias
    double bias = -50.0;

    cout << "What's the height: ";
    cin >> x1;
    cout << "What's the weight: ";
    cin >> x2;

    //Sigmoid function
    double X = (x1 * w1 + x2 * w2) + bias;
    double sigmoid = Sigmoid(X);
    
    if (sigmoid < 0.5) {
        cout << "This is not a human";
    }
    else if (sigmoid > 0.5) {
        cout << "This is a human";
    }

}

/*Research:

1)Binary Step Function:
  ->Decides whether a neuron is activated or not
  ->It can be False or True If the activation is <=0 or >0
  ->It can't be used for multivalue or classification problems since it's binary(0 or 1)
  ->It's gradient is 0 which is an hindrance for back propagation process

2)ReLU (Rectified Linear Unit):
  ->Returns 0 when the activation value is negative
  ->Returns the activation value itself when it is positive
  ->It introduces non-linearity into the neural network
  ->It is computationally simple and fast to calculate
  ->It helps reduce the vanishing gradient problem compared to Sigmoid
  ->It is one of the most commonly used activation functions in modern neural networks

3)Sigmoid Function:
  ->Converts the activation value into a value between 0 and 1
  ->It is useful for binary classification problems
  ->The output can be interpreted as a probability/confidence score
  ->It produces smooth and continuous outputs unlike the Binary Step Function
  ->Large positive values move towards 1 while large negative values move towards 0
  ->It can be used with backpropagation because it has a differentiable gradient

*/