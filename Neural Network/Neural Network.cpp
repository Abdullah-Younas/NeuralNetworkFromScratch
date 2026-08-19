// Neural Network.cpp
// Milestone 6: Train Neural Network On Real Dataset
// TrainingDataHumanClassifier

#include <iostream>
#include <filesystem>
#include <fstream>
#include <math.h>
#include <numbers>
#include <algorithm>

using namespace std;

double Sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

int main()
{

    //Training Data
    ifstream file("TrainingDataHumanClassifier.txt");

    if (!file)
    {
        cout << std::filesystem::current_path() << endl;
        cout << "Failed to open file!" << endl;
        return 1;
    }

    double height, weight, age, targett;
    file >> height >> weight >> age >> targett;

    //Loss function
    double target = targett; // not human
    double loss;
    double output_Unit_Error;
    double h1_Unit_Error;
    double h2_Unit_Error;
    double h3_Unit_Error;

    //Delta of Input weights
    double delta_w11;
    double delta_w12;
    double delta_w13;
    double delta_w21;
    double delta_w22;
    double delta_w23;
    double delta_w31;
    double delta_w32;
    double delta_w33;

    //Delta of Hidden weights
    double delta_hv1;
    double delta_hv2;
    double delta_hv3;

    //Learning rate
    double learning_Rate = 0.2;

    //Inputs
    double x1 = height; //Height in meter 0-1
    double x2 = weight; //Weight in kg 
    double x3 = age; //age in years
    
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
    double ob = -0.5;

    //Initalize layers
    double h1;
    double h2;
    double h3;
    double Sh1;
    double Sh2;
    double Sh3;

    //Output
    double output;
    double So;

    for (int epoch = 0; epoch <= 1000; epoch++) {

        //Sigmoid function
        h1 = (x1_Norm * w11 + x2_Norm * w12 + x3_Norm * w13 + hb1);
        h2 = (x1_Norm * w21 + x2_Norm * w22 + x3_Norm * w23 + hb1);
        h3 = (x1_Norm * w31 + x2_Norm * w32 + x3_Norm * w33 + hb1);
        Sh1 = Sigmoid(h1); //Sigmoid of hidden layer
        Sh2 = Sigmoid(h2);
        Sh3 = Sigmoid(h3);

        output = (Sh1 * v1 + Sh2 * v2 + Sh3 * v3 + ob);
        So = Sigmoid(output); //Sigmoid of output
        loss = target - So;

        //Iterations to train the network
        if (epoch % 100 == 0) {
            cout << "Epoch: " << epoch << " Prediction: " << So << " MSE: " << pow(loss, 2) << endl;
        }

        //Output unit error
        output_Unit_Error = So * (1.0 - So) * loss;
        //Hidden unit error
        h1_Unit_Error = Sh1 * (1 - Sh1) * (v1 * output_Unit_Error);
        h2_Unit_Error = Sh2 * (1 - Sh2) * (v2 * output_Unit_Error);
        h3_Unit_Error = Sh3 * (1 - Sh3) * (v3 * output_Unit_Error);
        //Delta of hidden layer to output layer
        delta_hv1 = learning_Rate * (output_Unit_Error)*Sh1;
        delta_hv2 = learning_Rate * (output_Unit_Error)*Sh2;
        delta_hv3 = learning_Rate * (output_Unit_Error)*Sh3;
        //Delta of Input layer to hidden layer
        delta_w11 = learning_Rate * (h1_Unit_Error)*x1;
        delta_w12 = learning_Rate * (h1_Unit_Error)*x1;
        delta_w13 = learning_Rate * (h1_Unit_Error)*x1;
        delta_w21 = learning_Rate * (h2_Unit_Error)*x2;
        delta_w22 = learning_Rate * (h2_Unit_Error)*x2;
        delta_w23 = learning_Rate * (h2_Unit_Error)*x2;
        delta_w31 = learning_Rate * (h3_Unit_Error)*x3;
        delta_w32 = learning_Rate * (h3_Unit_Error)*x3;
        delta_w33 = learning_Rate * (h3_Unit_Error)*x3;
        //Updating weights of hidden to output layers
        v1 = delta_hv1 + v1;
        v2 = delta_hv2 + v2;
        v3 = delta_hv3 + v3;
        //Updating weights of input to hidden layers
        w11 = delta_w11 + w11;
        w12 = delta_w12 + w12;
        w13 = delta_w13 + w13;
        w21 = delta_w21 + w21;
        w22 = delta_w22 + w22;
        w23 = delta_w23 + w23;
        w31 = delta_w31 + w31;
        w32 = delta_w32 + w32;
        w33 = delta_w33 + w33;

    }

}