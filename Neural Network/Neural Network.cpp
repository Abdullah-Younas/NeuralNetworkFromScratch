#include <iostream>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <vector>

using namespace std;

double Sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

int main()
{
    // Load weights

    ifstream file2("Weights.txt");

    if (!file2)
    {
        cout << filesystem::current_path() << endl;
        cout << "Failed to open Weights.txt!" << endl;
        return 1;
    }

    vector<double> Nweights;
    double value;

    while (file2 >> value)
    {
        Nweights.push_back(value);
    }

    file2.close();

    if (Nweights.size() < 12)
    {
        cout << "Weights.txt must contain at least 12 weights!" << endl;
        return 1;
    }

    // Input -> Hidden weights
    double w11 = Nweights[0];
    double w12 = Nweights[1];
    double w13 = Nweights[2];

    double w21 = Nweights[3];
    double w22 = Nweights[4];
    double w23 = Nweights[5];

    double w31 = Nweights[6];
    double w32 = Nweights[7];
    double w33 = Nweights[8];

    // Hidden -> Output weights
    double v1 = Nweights[9];
    double v2 = Nweights[10];
    double v3 = Nweights[11];

    // Biases

    double hb1 = -0.5;
    double ob = -0.5;

    // Learning rate
    double learning_Rate = 0.2;

    // Training
    const int epochs = 1000;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        // Open the dataset at the beginning
        // of every epoch.
        ifstream file("TrainingDataHumanClassifier.txt");

        if (!file)
        {
            cout << filesystem::current_path() << endl;
            cout << "Failed to open TrainingDataHumanClassifier.txt!" << endl;
            return 1;
        }

        double totalLoss = 0.0;
        int dataCount = 0;

        // Go through EVERY person

        double height;
        double weight;
        double age;
        double target;

        char comma;

        while (file >> height >> comma
            >> weight >> comma
            >> age >> comma
            >> target)
        {
            // Normalize inputs

            double x1 = (height - 0.5) / (2.5 - 0.5);
            double x2 = (weight - 3.0) / (120.0 - 3.0);
            double x3 = age / 100.0;

            // Forward pass

            double h1 =
                x1 * w11 +
                x2 * w12 +
                x3 * w13 +
                hb1;

            double h2 =
                x1 * w21 +
                x2 * w22 +
                x3 * w23 +
                hb1;

            double h3 =
                x1 * w31 +
                x2 * w32 +
                x3 * w33 +
                hb1;

            double Sh1 = Sigmoid(h1);
            double Sh2 = Sigmoid(h2);
            double Sh3 = Sigmoid(h3);

            double output =
                Sh1 * v1 +
                Sh2 * v2 +
                Sh3 * v3 +
                ob;

            double So = Sigmoid(output);

            // Loss

            double loss = target - So;

            totalLoss += loss * loss;
            dataCount++;

            // Backpropagation

            double output_Unit_Error =
                So * (1.0 - So) * loss;

            double h1_Unit_Error =
                Sh1 * (1.0 - Sh1) *
                (v1 * output_Unit_Error);

            double h2_Unit_Error =
                Sh2 * (1.0 - Sh2) *
                (v2 * output_Unit_Error);

            double h3_Unit_Error =
                Sh3 * (1.0 - Sh3) *
                (v3 * output_Unit_Error);

            // Hidden -> Output

            double delta_v1 =
                learning_Rate *
                output_Unit_Error *
                Sh1;

            double delta_v2 =
                learning_Rate *
                output_Unit_Error *
                Sh2;

            double delta_v3 =
                learning_Rate *
                output_Unit_Error *
                Sh3;

            // Input -> Hidden

            double delta_w11 =
                learning_Rate *
                h1_Unit_Error *
                x1;

            double delta_w12 =
                learning_Rate *
                h1_Unit_Error *
                x2;

            double delta_w13 =
                learning_Rate *
                h1_Unit_Error *
                x3;


            double delta_w21 =
                learning_Rate *
                h2_Unit_Error *
                x1;

            double delta_w22 =
                learning_Rate *
                h2_Unit_Error *
                x2;

            double delta_w23 =
                learning_Rate *
                h2_Unit_Error *
                x3;


            double delta_w31 =
                learning_Rate *
                h3_Unit_Error *
                x1;

            double delta_w32 =
                learning_Rate *
                h3_Unit_Error *
                x2;

            double delta_w33 =
                learning_Rate *
                h3_Unit_Error *
                x3;

            // Update weights

            v1 += delta_v1;
            v2 += delta_v2;
            v3 += delta_v3;

            w11 += delta_w11;
            w12 += delta_w12;
            w13 += delta_w13;

            w21 += delta_w21;
            w22 += delta_w22;
            w23 += delta_w23;

            w31 += delta_w31;
            w32 += delta_w32;
            w33 += delta_w33;
        }

        file.close();

        // Print epoch information

        if (epoch % 10 == 0)
        {
            double averageLoss = totalLoss / dataCount;

            cout << "Epoch: "
                << epoch
                << " Average MSE: "
                << averageLoss
                << endl;
        }
    }

    // Save updated weights

    ofstream weightFile("Weights.txt");

    if (!weightFile)
    {
        cout << "Failed to save Weights.txt!" << endl;
        return 1;
    }

    weightFile << w11 << endl;
    weightFile << w12 << endl;
    weightFile << w13 << endl;

    weightFile << w21 << endl;
    weightFile << w22 << endl;
    weightFile << w23 << endl;

    weightFile << w31 << endl;
    weightFile << w32 << endl;
    weightFile << w33 << endl;

    weightFile << v1 << endl;
    weightFile << v2 << endl;
    weightFile << v3 << endl;

    weightFile.close();

    cout << "Training complete!" << endl;
    cout << "Updated weights saved to Weights.txt" << endl;

    return 0;
}