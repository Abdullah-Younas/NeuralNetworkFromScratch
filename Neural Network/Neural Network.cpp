// Neural Network.cpp : This file contains the 'main' function. Program execution begins and ends there.
//Milestone 3: Loss Functions

//Loss function is basically the output - true(value)
//e.g; if we have 2 outputs 0 is Not human 1 is human and the output is 0.25 then the loss(e) will be e = output - true(1)

#include <iostream>
#include <math.h>
#include <numbers>
#include <algorithm>

using namespace std;

double Sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

int main()
{
    //Loss function
    double target = 1; // not human
    double loss;

    //Inputs
    double x1 = 1.89; //Height in meter 0-1
    double x2 = 60; //Weight in kg 
    double x3 = 15; //age in years
    
    double x1_Norm = (x1 - 0.5) / (2.5 - 0.5);
    double x2_Norm = (x2 - 3) / (120 - 3);
    double x3_Norm = x3 / 100.0;

    //Weights
    double w11 = 0.5; // Height
    double w12 = 0.9; // Weight
    double w13 = 0.6; // Age
    double w21 = 0.4; // Height
    double w22 = 0.8; // Weight
    double w23 = 0.7; // Age
    double w31 = 0.7; // Height
    double w32 = 0.5; // Weight
    double w33 = 0.6; // Age

    //Weights from hidden to output
    double v1 = 0.7;
    double v2 = 0.4;
    double v3 = 0.9;

    //Bias h = hidden layer o = output layer
    double hb1 = -0.5;
    double hb2 = 0.2;
    double hb3 = -0.3;
    double ob = -0.5;

    //Sigmoid function
    //h is hidden layer
    double h1 = (x1_Norm * w11 + x2_Norm * w12 + x3_Norm * w13 + hb1);
    double h2 = (x1_Norm * w21 + x2_Norm * w22 + x3_Norm * w23 + hb2);
    double h3 = (x1_Norm * w31 + x2_Norm * w32 + x3_Norm * w33 + hb3);
    double Sh1 = Sigmoid(h1); //Sigmoid of hidden layer
    double Sh2 = Sigmoid(h2); 
    double Sh3 = Sigmoid(h3);

    double output = (Sh1 * v1 + Sh2 * v2 + Sh3 * v3 + ob);
    double So = Sigmoid(output); //Sigmoid of output
   
    cout << "Prediction: " << So << endl;
    loss = target - So;
    cout << "Error: " << loss << endl;
    cout << "MSE: " << pow(loss, 2) << endl;

    if (So < 0.5) {
        cout << "Prediction(based on guessed weights): Not Human";
    }
    else if (So > 0.5) {
        cout << "Prediction(based on guessed weights): Human";
    }
}