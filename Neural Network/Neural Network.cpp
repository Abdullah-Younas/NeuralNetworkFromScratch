#include <iostream>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <vector>

using namespace std;
using std::cout;
using std::endl;
using std::cin;

vector<vector<double>> weightsInputHidden;
vector<double> weightsHiddenOutput;

vector<double> biasesHidden;
double biasOutput;

double Sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

vector<double> LoadWeights(string FileName, int input, int hidden) {
    ifstream file(FileName);

    if (!file)
    {
        cout << filesystem::current_path() << endl;
        cout << "Failed to open Weights.txt!" << endl;
        return {};
    }

    vector<double> Nweights;
    double value;

    while (file >> value)
    {
        Nweights.push_back(value);
    }

    file.close();

    if (Nweights.size() < (input * hidden + hidden))
    {
        cout << "Weights.txt does not contain enough weights!" << endl;
        return {};
    }

    weightsInputHidden.resize(hidden, vector<double>(input));
    int z = 0;

    for (int x = 0; x < hidden; x++) {
        for (int y = 0; y < input; y++) {
            weightsInputHidden[x][y] = Nweights[z];
            z++;
        }
    }

    weightsHiddenOutput.resize(hidden);

    for (int i = 0; i < hidden; i++) {
        weightsHiddenOutput[i] = Nweights[z];
        z++;
    }

    return Nweights;
}

void LoadBiases(string FileName, int hidden) {
    ifstream file(FileName);

    if (!file) {
        cout << filesystem::current_path() << endl;
        cout << "Failed to open Biases.txt!" << endl;
        return;
    }

    double value;
    vector<double> biases;

    while (file >> value) {
        biases.push_back(value);
    }

    file.close();

    if (biases.size() < hidden + 1) {
        cout << "Biases.txt must contain at least " << hidden + 1 << " biases!" << endl;
        return;
    }

    biasesHidden.resize(hidden);

    for (int h = 0; h < hidden; h++) {
        biasesHidden[h] = biases[h];
    }

    biasOutput = biases[hidden];
}

void TrainingNeuralNetwork(
    int NumOfEpochs,
    double LearningRate,
    vector<vector<double>>& weightsInputHidden,
    vector<double>& weightsHiddenOutput,
    int inputLayerSize,
    int hiddenLayerSize,
    vector<double>& biasesHidden,
    double& biasOutput)
{
    for (int epoch = 0; epoch < NumOfEpochs; epoch++)
    {
        ifstream file("TrainingDataHumanClassifier.txt");

        if (!file)
        {
            cout << filesystem::current_path() << endl;
            cout << "Failed to open TrainingDataHumanClassifier.txt!" << endl;
            return;
        }

        double totalLoss = 0.0;
        int dataCount = 0;

        double height;
        double weight;
        double age;
        double target;

        char comma;

        vector<double> hidden(hiddenLayerSize);
        vector<double> hiddenSigmoid(hiddenLayerSize);
        vector<double> hidden_Unit_Error(hiddenLayerSize);

        while (file >> height >> comma
            >> weight >> comma
            >> age >> comma
            >> target)
        {
            // Normalize inputs

            double x1 = (height - 0.5) / (2.5 - 0.5);
            double x2 = (weight - 3.0) / (120.0 - 3.0);
            double x3 = age / 100.0;

            vector<double> inputs = { x1, x2, x3 };

            // Forward pass

            for (int h = 0; h < hiddenLayerSize; h++)
            {
                hidden[h] = 0.0;

                for (int i = 0; i < inputLayerSize; i++)
                {
                    hidden[h] +=
                        inputs[i] *
                        weightsInputHidden[h][i];
                }

                hidden[h] += biasesHidden[h];

                hiddenSigmoid[h] = Sigmoid(hidden[h]);
            }

            double output = biasOutput;

            for (int h = 0; h < hiddenLayerSize; h++)
            {
                output +=
                    hiddenSigmoid[h] *
                    weightsHiddenOutput[h];
            }

            double So = Sigmoid(output);

            // Loss

            double loss = target - So;

            totalLoss += loss * loss;
            dataCount++;

            // Backpropagation

            double output_Unit_Error =
                So * (1.0 - So) * loss;

            for (int h = 0; h < hiddenLayerSize; h++)
            {
                hidden_Unit_Error[h] =
                    hiddenSigmoid[h] *
                    (1.0 - hiddenSigmoid[h]) *
                    (weightsHiddenOutput[h] *
                        output_Unit_Error);
            }

            // Hidden -> Output

            for (int h = 0; h < hiddenLayerSize; h++)
            {
                double delta_v =
                    LearningRate *
                    output_Unit_Error *
                    hiddenSigmoid[h];

                weightsHiddenOutput[h] += delta_v;
            }
            double delta_output_bias = LearningRate * output_Unit_Error;
            biasOutput += delta_output_bias;

            // Input -> Hidden

            for (int h = 0; h < hiddenLayerSize; h++)
            {
                for (int i = 0; i < inputLayerSize; i++)
                {
                    double delta_w =
                        LearningRate *
                        hidden_Unit_Error[h] *
                        inputs[i];

                    weightsInputHidden[h][i] += delta_w;
                }
            }
            for (int h = 0; h < hiddenLayerSize; h++) {
                double delta_hidden_bias = LearningRate * hidden_Unit_Error[h];
                biasesHidden[h] += delta_hidden_bias;
            }
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
}

void PredictingMode(int inputLayerSize, int hiddenLayerSize, vector<double>& biasesHidden, double& biasOutput)
{
    double height;
    double weight;
    double age;

    cout << "Enter the Height, Weight, Age (in that exact order, separated by spaces): ";
    cin >> height >> weight >> age;

    double x1 = (height - 0.5) / (2.5 - 0.5);
    double x2 = (weight - 3.0) / (120.0 - 3.0);
    double x3 = age / 100.0;

    vector<double> inputs = { x1, x2, x3 };

    vector<double> hidden(hiddenLayerSize);
    vector<double> hiddenSigmoid(hiddenLayerSize);

    for (int h = 0; h < hiddenLayerSize; h++)
    {
        hidden[h] = 0.0;

        for (int i = 0; i < inputLayerSize; i++)
        {
            hidden[h] +=
                inputs[i] *
                weightsInputHidden[h][i];
        }

        hidden[h] += biasesHidden[h];

        hiddenSigmoid[h] = Sigmoid(hidden[h]);
    }

    double output = biasOutput;

    for (int h = 0; h < hiddenLayerSize; h++)
    {
        output +=
            hiddenSigmoid[h] *
            weightsHiddenOutput[h];
    }

    double So = Sigmoid(output);

    if (So >= 0.5)
    {
        cout << "Prediction: Human" << endl;
    }
    else
    {
        cout << "Prediction: Not Human" << endl;
    }

    cout << "Probability: " << So << endl;
}

int main()
{
    // Layers

    int input = 3;
    int hidden = 9;
    int output = 1;

    int Mode;


    // Load weights
    LoadWeights("Weights.txt", input, hidden);

    // Biases
    LoadBiases("Biases.txt", hidden);

    // Learning rate
    double learning_Rate = 0.05;

    // Training
    const int epochs = 1000;
    
    cout << "Choose a Mode(Press 0 for Training the Network OR Press 1 for Predicting): ";
    cin >> Mode;

    if (Mode == 0) {

        TrainingNeuralNetwork(
            epochs,
            learning_Rate,
            weightsInputHidden,
            weightsHiddenOutput,
            input,
            hidden,
            biasesHidden,
            biasOutput
        );

        // Save updated weights

        ofstream weightFile("Weights.txt");

        if (!weightFile)
        {
            std::cout << "Failed to save Weights.txt!" << endl;
            return 1;
        }

        for (int h = 0; h < hidden; h++)
        {
            for (int i = 0; i < input; i++)
            {
                weightFile << weightsInputHidden[h][i] << endl;
            }
        }

        for (int h = 0; h < hidden; h++)
        {
            weightFile << weightsHiddenOutput[h] << endl;
        }

        weightFile.close();

        cout << "Training complete!" << endl;
        cout << "Updated weights saved to Weights.txt" << endl;

        // Save updated Biases

        ofstream biasFile("Biases.txt");

        if (!biasFile)
        {
            cout << "Failed to save Biases.txt!" << endl;
            return 1;
        }

        for (int h = 0; h < hidden; h++)
        {
            biasFile << biasesHidden[h] << endl;
        }

        biasFile << biasOutput << endl;

        biasFile.close();

        cout << "Updated biases saved to Biases.txt" << endl;

    }
    else if (Mode == 1) {
        PredictingMode(input, hidden, biasesHidden, biasOutput);
    }
    else {
        cout << "Unknown Mode";
    }



    return 0;
}