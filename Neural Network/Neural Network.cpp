// Neural Network.cpp : This file contains the 'main' function. Program execution begins and ends there.
//Milestone 3: Forward pass through a neural network with one hidden layer.
//Weights and biases are manually chosen and not trained.

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
    double x3; //age in years

    //Weights
    double w11 = 1.0; // Height
    double w12 = 2.0; // Weight
    double w13 = 3.0; // Age
    double w21 = 1.0; // Height
    double w22 = 2.0; // Weight
    double w23 = 3.0; // Age
    double w31 = 1.0; // Height
    double w32 = 2.0; // Weight
    double w33 = 3.0; // Age

    //Weights from hidden to output
    double v1 = 2.0;
    double v2 = 2.0;
    double v3 = 2.0;

    //Bias h = hidden layer o = output layer
    double hb1 = -50.0;
    double hb2 = -100.0;
    double hb3 = 0.0;
    double ob = 50.0;

    cout << "What's the height: ";
    cin >> x1;
    cout << "What's the weight: ";
    cin >> x2;
    cout << "What's the age: ";
    cin >> x3;

    //Sigmoid function
    //h is hidden layer
    double h1 = (x1 * w11 + x2 * w12 + x3 * w13 + hb1);
    double h2 = (x1 * w21 + x2 * w22 + x3 * w23 + hb2);
    double h3 = (x1 * w31 + x2 * w32 + x3 * w33 + hb3);
    double Sh1 = Sigmoid(h1); //Sigmoid of hidden layer
    double Sh2 = Sigmoid(h2); 
    double Sh3 = Sigmoid(h3);

    double output = (Sh1 * v1 + Sh2 * v2 + Sh3 * v3 + ob);
    double So = Sigmoid(output); //Sigmoid of output
   
    cout << "Final Sigmoid: " << So << endl;

    if (So < 0.5) {
        cout << "Prediction(based on guessed weights): Not Human";
    }
    else if (So > 0.5) {
        cout << "Prediction(based on guessed weights): Human";
    }
}