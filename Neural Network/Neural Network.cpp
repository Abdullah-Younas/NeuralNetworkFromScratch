#include <iostream>
#include <random>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <vector>
#include <cstdint> 

using namespace std;

const int OutputLayers = 10;

vector<vector<vector<float>>> TrainingImagesData;
vector<int> TrainingImagesLabels;

vector<vector<float>> C1Filter1(5, vector<float>(5));
vector<vector<float>> C1Filter2(5, vector<float>(5));

vector<vector<vector<vector<float>>>> C2Filters(4, vector<vector<vector<float>>>(2,vector<vector<float>>(3,vector<float>(3))));

vector<vector<float>> C1Image1(24, vector<float>(24));
vector<vector<float>> C1Image2(24, vector<float>(24));

vector<vector<float>> C1Image1Pooled(12, vector<float>(12));
vector<vector<float>> C1Image2Pooled(12, vector<float>(12));

vector<vector<float>> C2Image1(10, vector<float>(10));
vector<vector<float>> C2Image2(10, vector<float>(10));
vector<vector<float>> C2Image3(10, vector<float>(10));
vector<vector<float>> C2Image4(10, vector<float>(10));

vector<vector<float>> C2Image1Pooled(5, vector<float>(5));
vector<vector<float>> C2Image2Pooled(5, vector<float>(5));
vector<vector<float>> C2Image3Pooled(5, vector<float>(5));
vector<vector<float>> C2Image4Pooled(5, vector<float>(5));

vector<float> FlatPixelValues(100);

vector<vector<float>> FlatToOutputWeights(10, vector<float>(100));
vector<float> OutputBiases(10);

vector<float> Target(10);
vector<float> Output(10);
vector<float> OutputError(OutputLayers);

vector<float> FlatError(100, 0.0f);

const float LearningRate = 0.05f;

float C1F1Bias;
float C1F2Bias;
float C2F1Bias;
float C2F2Bias;
float C2F3Bias;
float C2F4Bias;

void LoadTrainingDataMNIST(int imgsToLoad)
{
    ifstream file("train-images-idx3-ubyte", ios::binary);

    if (!file.is_open())
    {
        cout << "Failed to open MNIST file!" << endl;
        return;
    }

    file.seekg(16); // Skip MNIST header

    int images = imgsToLoad;

    TrainingImagesData.resize(images, vector<vector<float>>(
        28, vector<float>(28)
    ));

    for (int i = 0; i < images; i++)
    {
        for (int r = 0; r < 28; r++)
        {
            for (int c = 0; c < 28; c++)
            {
                unsigned char pixel;
                file.read((char*)&pixel, 1);

                TrainingImagesData[i][r][c] =
                    pixel / 255.0f;
            }
        }
    }
}

void loadTrainingDataLabels(int Labels)
{
    ifstream file("train-labels-idx1-ubyte", ios::binary);

    if (!file.is_open())
    {
        cout << "Failed to open MNIST label file!" << endl;
        return;
    }

    cout << "MNIST Label file opened!" << endl;

    file.seekg(8); // Skip MNIST label header

    TrainingImagesLabels.resize(Labels);

    for (int i = 0; i < Labels; i++)
    {
        unsigned char label;
        file.read((char*)&label, 1);

        TrainingImagesLabels[i] = label;
    }
}

void LoadRandomWeightsIntoC1Filters()
{
    ifstream inFile("C1Weights.txt");

    // File exists and has data
    if (inFile.good() && inFile.peek() != EOF)
    {
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                inFile >> C1Filter1[i][j];
            }
        }

        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                inFile >> C1Filter2[i][j];
            }
        }

        return;
    }

    // File missing OR empty
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-1.0f, 1.0f);

    ofstream outFile("C1Weights.txt");

    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            C1Filter1[i][j] = dis(gen);
            outFile << C1Filter1[i][j] << '\n';
        }
    }

    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            C1Filter2[i][j] = dis(gen);
            outFile << C1Filter2[i][j] << '\n';
        }
    }

}

void LoadRandomWeightsIntoC2Filters()
{
    ifstream inFile("C2Weights.txt");

    // File exists and is not empty
    if (inFile.good() && inFile.peek() != EOF)
    {
        // Load existing weights
        for (int filters = 0; filters < 4; filters++)
        {
            for (int channel = 0; channel < 2; channel++)
            {
                for (int i = 0; i < 3; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        inFile >> C2Filters[filters][channel][i][j];
                    }
                }
            }
        }

        inFile.close();

        return;
    }

    // File doesn't exist OR is empty
    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<float> dis(-1.0f, 1.0f);

    ofstream outFile("C2Weights.txt");

    // Generate and save random weights
    for (int filters = 0; filters < 4; filters++)
    {
        for (int channel = 0; channel < 2; channel++)
        {
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    C2Filters[filters][channel][i][j] = dis(gen);

                    outFile << C2Filters[filters][channel][i][j] << '\n';
                }
            }
        }
    }

    outFile.close();

}

void LoadRandomWeightsIntoFlatToOutputWeights()
{
    ifstream inFile("DenseWeights.txt");

    // File exists and contains data
    if (inFile.good() && inFile.peek() != EOF)
    {
        // Load existing weights
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 100; j++)
            {
                inFile >> FlatToOutputWeights[i][j];
            }
        }

        inFile.close();

        return;
    }

    // File doesn't exist OR is empty
    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<float> dis(-1.0f, 1.0f);

    ofstream outFile("DenseWeights.txt");

    // Generate and save random weights
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 100; j++)
        {
            FlatToOutputWeights[i][j] = dis(gen);

            outFile << FlatToOutputWeights[i][j] << '\n';
        }
    }

    outFile.close();

}

void LoadC1C2Biases()
{
    ifstream inFile("C1C2Biases.txt");

    if (inFile.good() && inFile.peek() != EOF)
    {
        inFile >> C1F1Bias;
        inFile >> C1F2Bias;
        inFile >> C2F1Bias;
        inFile >> C2F2Bias;
        inFile >> C2F3Bias;
        inFile >> C2F4Bias;

        inFile.close();

        return;
    }
}

void LoadRandomBiasesIntoOutputLayer()
{
    ifstream inFile("OutputBiases.txt");

    // File exists and contains data
    if (inFile.good() && inFile.peek() != EOF)
    {
        for (int i = 0; i < 10; i++)
        {
            inFile >> OutputBiases[i];
        }

        inFile.close();

        return;
    }

    // File doesn't exist OR is empty
    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<float> dis(-1.0f, 1.0f);

    ofstream outFile("OutputBiases.txt");

    for (int i = 0; i < 10; i++)
    {
        OutputBiases[i] = dis(gen);

        outFile << OutputBiases[i] << '\n';
    }

    outFile.close();

}

float Sigmoid(float sum)
{
    return static_cast<float>(1.0 / (1.0 + exp(-sum)));
}

float ReLU(float sum) {
    float value = 0;
    if (sum < 0) {
        value = 0;
    }
    else if (sum >= 0) {
        value = sum;
    } 
    return sum = value;
}

vector<float> Softmax(vector<float> Logits)
{
    float MaxLogit = *max_element(Logits.begin(), Logits.end());

    float TotalSum = 0.0f;

    for (int i = 0; i < Logits.size(); i++) {
        Logits[i] = exp(Logits[i] - MaxLogit);
        TotalSum += Logits[i];
    }

    for (int i = 0; i < Logits.size(); i++) {
        Logits[i] /= TotalSum;
    }

    return Logits;
}

void TrainCNN() {

    int epochs = 10;

    LoadTrainingDataMNIST(60000);
    loadTrainingDataLabels(60000);
    LoadRandomWeightsIntoC1Filters();
    LoadRandomWeightsIntoC2Filters();
    LoadRandomWeightsIntoFlatToOutputWeights();
    LoadRandomBiasesIntoOutputLayer();
    LoadC1C2Biases();

    //Convolution 1
    int C1rows = 28;
    int C1cols = 28;
    int C1kernelSize = 5;

    //Pooling
    int PoolSize = 2;
    int PoolStride = 2;

    //Convolution 2
    int C2rows = 12;
    int C2cols = 12;
    int C2KernelSize = 3;

    //Flattening Pooled images into a single vector/array

    for (int e = 0; e < epochs; e++) {
        
        cout << "Epoch " << e << ": " << endl;

        for (int img = 0; img < 60000; img++) {
        
            int FAIndex = 0;

            //Convolution 1 Kernel Processed Images
            for (int i = 0; i <= C1rows - C1kernelSize; i++) {
                for (int j = 0; j <= C1cols - C1kernelSize; j++) {

                    float sum1 = 0.0f;
                    float sum2 = 0.0f;

                    for (int k = 0; k < C1kernelSize; k++) {
                        for (int l = 0; l < C1kernelSize; l++) {

                            sum1 += TrainingImagesData[img][i + k][j + l] * C1Filter1[k][l];

                            sum2 += TrainingImagesData[img][i + k][j + l] * C1Filter2[k][l];

                        }
                    }
                    C1Image1[i][j] = ReLU(sum1 + C1F1Bias);
                    C1Image2[i][j] = ReLU(sum2 + C1F2Bias);
                }
            }

            //Convolution 1 Pooled Images
            for (int i = 0; i <= 24 - PoolSize; i += PoolStride) {
                for (int j = 0; j <= 24 - PoolSize; j += PoolStride) {
                    float TempMax1 = C1Image1[i][j];
                    float TempMax2 = C1Image2[i][j];

                    for (int k = 0; k < PoolSize; k++) {
                        for (int l = 0; l < PoolSize; l++) {
                            if (C1Image1[i + k][j + l] > TempMax1) {
                                TempMax1 = C1Image1[i + k][j + l];
                            }
                            if (C1Image2[i + k][j + l] > TempMax2) {
                                TempMax2 = C1Image2[i + k][j + l];
                            }
                        }
                    }
                    C1Image1Pooled[i / PoolStride][j / PoolStride] = TempMax1;
                    C1Image2Pooled[i / PoolStride][j / PoolStride] = TempMax2;
                }
            }

            //Convolution 2 Kernel Processed Images
            for (int filter = 0; filter < 4; filter++) {

                for (int i = 0; i <= C2rows - C2KernelSize; i++) {
                    for (int j = 0; j <= C2cols - C2KernelSize; j++) {

                        float sum = 0.0f;

                        for (int channel = 0; channel < 2; channel++) {

                            for (int k = 0; k < C2KernelSize; k++) {
                                for (int l = 0; l < C2KernelSize; l++) {

                                    if (channel == 0) {
                                        sum += C1Image1Pooled[i + k][j + l]
                                            * C2Filters[filter][channel][k][l];
                                    }
                                    else {
                                        sum += C1Image2Pooled[i + k][j + l]
                                            * C2Filters[filter][channel][k][l];
                                    }

                                }
                            }
                        }
                        if (filter == 0) C2Image1[i][j] = Sigmoid(sum + C2F1Bias);
                        if (filter == 1) C2Image2[i][j] = Sigmoid(sum + C2F2Bias);
                        if (filter == 2) C2Image3[i][j] = Sigmoid(sum + C2F3Bias);
                        if (filter == 3) C2Image4[i][j] = Sigmoid(sum + C2F4Bias);
                    }
                }
            }

            //Convolution 2 Pooled Images
            for (int i = 0; i <= 10 - PoolSize; i += PoolStride) {
                for (int j = 0; j <= 10 - PoolSize; j += PoolStride) {
                    float TempMax1 = C2Image1[i][j];
                    float TempMax2 = C2Image2[i][j];
                    float TempMax3 = C2Image3[i][j];
                    float TempMax4 = C2Image4[i][j];

                    for (int k = 0; k < PoolSize; k++) {
                        for (int l = 0; l < PoolSize; l++) {
                            if (C2Image1[i + k][j + l] > TempMax1) {
                                TempMax1 = C2Image1[i + k][j + l];
                            }
                            if (C2Image2[i + k][j + l] > TempMax2) {
                                TempMax2 = C2Image2[i + k][j + l];
                            }
                            if (C2Image3[i + k][j + l] > TempMax3) {
                                TempMax3 = C2Image3[i + k][j + l];
                            }
                            if (C2Image4[i + k][j + l] > TempMax4) {
                                TempMax4 = C2Image4[i + k][j + l];
                            }
                        }
                    }
                    C2Image1Pooled[i / PoolStride][j / PoolStride] = TempMax1;
                    C2Image2Pooled[i / PoolStride][j / PoolStride] = TempMax2;
                    C2Image3Pooled[i / PoolStride][j / PoolStride] = TempMax3;
                    C2Image4Pooled[i / PoolStride][j / PoolStride] = TempMax4;
                }
            }

            //Flattening pooled images into a single vector 
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    FlatPixelValues[FAIndex++] = C2Image1Pooled[i][j];
                }
            }
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    FlatPixelValues[FAIndex++] = C2Image2Pooled[i][j];
                }
            }
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    FlatPixelValues[FAIndex++] = C2Image3Pooled[i][j];
                }
            }
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    FlatPixelValues[FAIndex++] = C2Image4Pooled[i][j];
                }
            }

            //Flat to Output Connection
            for (int i = 0; i < OutputLayers; i++) {
                Output[i] = 0;
                for (int j = 0; j < 100; j++) {
                    Output[i] += FlatPixelValues[j] * FlatToOutputWeights[i][j];
                }
                Output[i] += OutputBiases[i];
            }

            vector<float> Probabilities = Softmax(Output);

            if(img % 5000 == 0){
                for (int i = 0; i < OutputLayers; i++){
                    cout << "Probabilities[" << i << "]: " << Probabilities[i] << endl;
                }
            }

            //Output Errors
            for (int i = 0; i < OutputLayers; i++) {
                if (i == TrainingImagesLabels[img])
                {
                    OutputError[i] = Probabilities[i] - 1.0f;
                }
                else
                {
                    OutputError[i] = Probabilities[i];
                }
            }

            //Reset flat Errors
            for (int j = 0; j < 100; j++)
            {
                FlatError[j] = 0.0f;
            }

            //Calculating Flat Errors
            for (int j = 0; j < 100; j++)
            {
                for (int i = 0; i < OutputLayers; i++)
                {
                    FlatError[j] +=
                        OutputError[i] *
                        FlatToOutputWeights[i][j];
                }
            }

            //Updating Flat to output weights
            for (int i = 0; i < OutputLayers; i++)
            {
                for (int j = 0; j < 100; j++)
                {
                    FlatToOutputWeights[i][j] -=
                        LearningRate *
                        OutputError[i] *
                        FlatPixelValues[j];
                }

                OutputBiases[i] -=
                    LearningRate *
                    OutputError[i];
            }

        }

        ofstream outFile("DenseWeights.txt");
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 100; j++)
            {
                outFile << FlatToOutputWeights[i][j] << '\n';
            }
        }

        ofstream BiasOutFile("OutputBiases.txt");
        for (int i = 0; i < OutputLayers; i++)
        {
            BiasOutFile << OutputBiases[i] << '\n';
        }
    }

}

int main()
{
    
    TrainCNN();

    return 0;
}

