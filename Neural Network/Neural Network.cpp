// Neural Network.cpp : This file contains the 'main' function. Program execution begins and ends there.
//Milestone 1: Create a perceptron that decides whether we'll go to the movies based on weather, company, proximity/Distance

#include <iostream>
#include <math.h>

using namespace std;


int main()
{
    //Inputs
    int x1; //weather
    int x2; //company
    int x3; //proximity

    //Weights
    int w1 = 4;
    int w2 = 2;
    int w3 = 2;

    //Bias
    int bias = -5;

    cout << "Weathers good?: ";
    cin >> x1;
    cout << "Company good?: ";
    cin >> x2;
    cout << "Proximity good?: ";
    cin >> x3;

    //Activation/Step function
    int activation = x1 * w1 + x2 * w2 + x3 * w3 + bias;
    
    if (activation <= 0) {
        cout << "You will not go to the movies";
    }
    else if (activation > 0) {
        cout << "You will go to the movies";
    }

}
