#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <iostream>
#include <random>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <vector>
#include <cstdint>
#include <algorithm>
#include "stb_image.h"
#include "stb_image_resize2.h"

using namespace std;

const int OutputLayers = 2;

vector<vector<float>> TrainingImagesData;
vector<int> TrainingLabels;

vector<vector<float>> C1Filter1(5, vector<float>(5));
vector<vector<float>> C1Filter2(5, vector<float>(5));

vector<vector<vector<vector<float>>>> C2Filters(
    4,
    vector<vector<vector<float>>>(
        2,
        vector<vector<float>>(3, vector<float>(3))
    )
);

vector<vector<float>> C1Image1(124, vector<float>(124));
vector<vector<float>> C1Image2(124, vector<float>(124));

vector<vector<float>> C1Image1Pooled(62, vector<float>(62));
vector<vector<float>> C1Image2Pooled(62, vector<float>(62));

vector<vector<float>> C2Image1(60, vector<float>(60));
vector<vector<float>> C2Image2(60, vector<float>(60));
vector<vector<float>> C2Image3(60, vector<float>(60));
vector<vector<float>> C2Image4(60, vector<float>(60));

vector<vector<float>> C2Image1Pooled(30, vector<float>(30));
vector<vector<float>> C2Image2Pooled(30, vector<float>(30));
vector<vector<float>> C2Image3Pooled(30, vector<float>(30));
vector<vector<float>> C2Image4Pooled(30, vector<float>(30));

vector<float> FlatPixelValues(3600);

vector<vector<float>> FlatToOutputWeights(2, vector<float>(3600));
vector<float> OutputBiases(2);

vector<float> Output(2);
vector<float> OutputError(2);
vector<float> FlatError(3600, 0.0f);

vector<vector<float>> C2PooledErrors(4, vector<float>(900, 0.0f));

vector<vector<float>> C2Error1(60, vector<float>(60, 0.0f));
vector<vector<float>> C2Error2(60, vector<float>(60, 0.0f));
vector<vector<float>> C2Error3(60, vector<float>(60, 0.0f));
vector<vector<float>> C2Error4(60, vector<float>(60, 0.0f));

vector<vector<float>> C1PooledError1(62, vector<float>(62, 0.0f));
vector<vector<float>> C1PooledError2(62, vector<float>(62, 0.0f));

vector<vector<float>> C1Error1(124, vector<float>(124, 0.0f));
vector<vector<float>> C1Error2(124, vector<float>(124, 0.0f));

vector<float> C2DLoss1(18, 0.0f);
vector<float> C2DLoss2(18, 0.0f);
vector<float> C2DLoss3(18, 0.0f);
vector<float> C2DLoss4(18, 0.0f);

vector<float> C1DLoss1(25, 0.0f);
vector<float> C1DLoss2(25, 0.0f);

float C1F1Bias = 0.0f;
float C1F2Bias = 0.0f;

float C2F1Bias = 0.0f;
float C2F2Bias = 0.0f;
float C2F3Bias = 0.0f;
float C2F4Bias = 0.0f;

int width, height, channels;

int tar_w = 128;
int tar_h = 128;

void LoadTrainingImagesCats() {
    for (const auto& file : filesystem::directory_iterator("Cats")) {
        string filename = file.path().string();

        unsigned char* img = stbi_load(filename.c_str(), &width, &height, &channels, 1);

        if (img) {
            unsigned char* output = (unsigned char*)malloc(tar_w * tar_h);

            stbir_resize_uint8_linear(img, width, height, 0, output, tar_w, tar_h, 0, STBIR_1CHANNEL);

            vector<float> image(16384);

            for (int i = 0; i < 16384; i++) {
                image[i] = output[i] / 255.0f;
            }
            

            TrainingImagesData.push_back(image);
            TrainingLabels.push_back(0);
           
            if (TrainingImagesData.size() % 1000 == 0)
            {
                cout << "Images loaded: " << TrainingImagesData.size() << endl;
            }


            free(output);
            stbi_image_free(img);
        }
    }
}

void LoadTrainingImagesDogs() {
    for (const auto& file : filesystem::directory_iterator("Dogs")) {
        string filename = file.path().string();

        unsigned char* img = stbi_load(filename.c_str(), &width, &height, &channels, 1);

        if (img) {
            unsigned char* output = (unsigned char*)malloc(tar_w * tar_h);

            stbir_resize_uint8_linear(img, width, height, 0, output, tar_w, tar_h, 0, STBIR_1CHANNEL);

            vector<float> image(16384);

            for (int i = 0; i < 16384; i++) {
                image[i] = output[i] / 255.0f;
            }

            TrainingImagesData.push_back(image);
            TrainingLabels.push_back(1);

            if (TrainingImagesData.size() % 1000 == 0)
            {
                cout << "Images loaded: " << TrainingImagesData.size() << endl;
            }

            free(output);
            stbi_image_free(img);
        }
    }
}

void LoadRandomWeightsIntoC1Filters()
{
    ifstream inFile("C1Weights.txt");

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

        inFile.close();

        return;
    }

    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<float> dis(-0.1f, 0.1f);

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

    outFile.close();
}

void LoadRandomWeightsIntoC2Filters()
{
    ifstream inFile("C2Weights.txt");

    if (inFile.good() && inFile.peek() != EOF)
    {
        for (int filter = 0; filter < 4; filter++)
        {
            for (int channel = 0; channel < 2; channel++)
            {
                for (int i = 0; i < 3; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        inFile >> C2Filters[filter][channel][i][j];
                    }
                }
            }
        }

        inFile.close();

        return;
    }

    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<float> dis(-0.1f, 0.1f);

    ofstream outFile("C2Weights.txt");

    for (int filter = 0; filter < 4; filter++)
    {
        for (int channel = 0; channel < 2; channel++)
        {
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    C2Filters[filter][channel][i][j] = dis(gen);

                    outFile << C2Filters[filter][channel][i][j] << '\n';
                }
            }
        }
    }

    outFile.close();
}

void LoadRandomWeightsIntoFlatToOutputWeights()
{
    ifstream inFile("DenseWeights.txt");

    if (inFile.good() && inFile.peek() != EOF)
    {
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3600; j++)
            {
                inFile >> FlatToOutputWeights[i][j];
            }
        }

        inFile.close();

        return;
    }

    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<float> dis(-0.1f, 0.1f);

    ofstream outFile("DenseWeights.txt");

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3600; j++)
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

    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<float> dis(-0.1f, 0.1f);

    C1F1Bias = dis(gen);
    C1F2Bias = dis(gen);

    C2F1Bias = dis(gen);
    C2F2Bias = dis(gen);
    C2F3Bias = dis(gen);
    C2F4Bias = dis(gen);
}

void LoadRandomBiasesIntoOutputLayer()
{
    ifstream inFile("OutputBiases.txt");

    if (inFile.good() && inFile.peek() != EOF)
    {
        for (int i = 0; i < 2; i++)
        {
            inFile >> OutputBiases[i];
        }

        inFile.close();

        return;
    }

    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<float> dis(-0.1f, 0.1f);

    ofstream outFile("OutputBiases.txt");

    for (int i = 0; i < 2; i++)
    {
        OutputBiases[i] = dis(gen);

        outFile << OutputBiases[i] << '\n';
    }

    outFile.close();
}

float ReLU(float value)
{
    if (value > 0.0f)
    {
        return value;
    }

    return 0.0f;
}

vector<float> Softmax(vector<float> Logits)
{
    float MaxLogit =
        *max_element(Logits.begin(), Logits.end());

    float TotalSum = 0.0f;

    for (int i = 0; i < Logits.size(); i++)
    {
        Logits[i] =
            exp(Logits[i] - MaxLogit);

        TotalSum += Logits[i];
    }

    for (int i = 0; i < Logits.size(); i++)
    {
        Logits[i] /= TotalSum;
    }

    return Logits;
}

void SaveWeights()
{
    ofstream DenseOut("DenseWeights.txt");

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3600; j++)
        {
            DenseOut << FlatToOutputWeights[i][j] << '\n';
        }
    }

    DenseOut.close();

    ofstream OutputBiasOut("OutputBiases.txt");

    for (int i = 0; i < 2; i++)
    {
        OutputBiasOut << OutputBiases[i] << '\n';
    }

    OutputBiasOut.close();

    ofstream C1Out("C1Weights.txt");

    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            C1Out << C1Filter1[i][j] << '\n';
        }
    }

    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            C1Out << C1Filter2[i][j] << '\n';
        }
    }

    C1Out.close();

    ofstream C2Out("C2Weights.txt");

    for (int filter = 0; filter < 4; filter++)
    {
        for (int channel = 0; channel < 2; channel++)
        {
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    C2Out <<
                        C2Filters[filter][channel][i][j]
                        << '\n';
                }
            }
        }
    }

    C2Out.close();

    ofstream BiasOut("C1C2Biases.txt");

    BiasOut << C1F1Bias << '\n';
    BiasOut << C1F2Bias << '\n';

    BiasOut << C2F1Bias << '\n';
    BiasOut << C2F2Bias << '\n';
    BiasOut << C2F3Bias << '\n';
    BiasOut << C2F4Bias << '\n';

    BiasOut.close();
}

void TrainCNN()
{
    int epochs = 10;

    LoadTrainingImagesCats();
    LoadTrainingImagesDogs();

    int images = TrainingImagesData.size();

    cout << "Total Images loaded: " << images << endl;
    cout << "Total Labels loaded: " << TrainingLabels.size() << endl;
    cout << "Image size: " << TrainingImagesData[0].size() << " pixels" << endl;
    cout << "Starting CNN training..." << endl;

    LoadRandomWeightsIntoC1Filters();
    LoadRandomWeightsIntoC2Filters();
    LoadRandomWeightsIntoFlatToOutputWeights();

    LoadRandomBiasesIntoOutputLayer();
    LoadC1C2Biases();

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float LearningRate = 0.05f;
        float TotalLoss = 0.0f;
        int Correct = 0;

        cout << "Starting Epoch " << epoch + 1 << "/" << epochs << endl;

        for (int img = 0; img < images; img++)
        {
            for (int i = 0; i < 124; i++)
            {
                for (int j = 0; j < 124; j++)
                {
                    C1Image1[i][j] = 0.0f;
                    C1Image2[i][j] = 0.0f;
                }
            }

            for (int i = 0; i < 62; i++)
            {
                for (int j = 0; j < 62; j++)
                {
                    C1Image1Pooled[i][j] = 0.0f;
                    C1Image2Pooled[i][j] = 0.0f;
                }
            }

            for (int i = 0; i < 60; i++)
            {
                for (int j = 0; j < 60; j++)
                {
                    C2Image1[i][j] = 0.0f;
                    C2Image2[i][j] = 0.0f;
                    C2Image3[i][j] = 0.0f;
                    C2Image4[i][j] = 0.0f;
                }
            }

            for (int i = 0; i < 30; i++)
            {
                for (int j = 0; j < 30; j++)
                {
                    C2Image1Pooled[i][j] = 0.0f;
                    C2Image2Pooled[i][j] = 0.0f;
                    C2Image3Pooled[i][j] = 0.0f;
                    C2Image4Pooled[i][j] = 0.0f;
                }
            }

            for (int i = 0; i < 124; i++)
            {
                for (int j = 0; j < 124; j++)
                {
                    float Sum1 = 0.0f;
                    float Sum2 = 0.0f;

                    for (int k = 0; k < 5; k++)
                    {
                        for (int l = 0; l < 5; l++)
                        {
                            Sum1 += TrainingImagesData[img][(i + k) * 128 + (j + l)] * C1Filter1[k][l];
                            Sum2 += TrainingImagesData[img][(i + k) * 128 + (j + l)] * C1Filter2[k][l];
                        }
                    }

                    C1Image1[i][j] = ReLU(Sum1 + C1F1Bias);
                    C1Image2[i][j] = ReLU(Sum2 + C1F2Bias);
                }
            }

            for (int i = 0; i < 62; i++)
            {
                for (int j = 0; j < 62; j++)
                {
                    int r = i * 2;
                    int c = j * 2;

                    float Max1 = C1Image1[r][c];
                    float Max2 = C1Image2[r][c];

                    for (int k = 0; k < 2; k++)
                    {
                        for (int l = 0; l < 2; l++)
                        {
                            if (C1Image1[r + k][c + l] > Max1)
                            {
                                Max1 = C1Image1[r + k][c + l];
                            }

                            if (C1Image2[r + k][c + l] > Max2)
                            {
                                Max2 = C1Image2[r + k][c + l];
                            }
                        }
                    }

                    C1Image1Pooled[i][j] = Max1;
                    C1Image2Pooled[i][j] = Max2;
                }
            }

            for (int filter = 0; filter < 4; filter++)
            {
                for (int i = 0; i < 60; i++)
                {
                    for (int j = 0; j < 60; j++)
                    {
                        float Sum = 0.0f;

                        for (int channel = 0; channel < 2; channel++)
                        {
                            for (int k = 0; k < 3; k++)
                            {
                                for (int l = 0; l < 3; l++)
                                {
                                    if (channel == 0)
                                    {
                                        Sum += C1Image1Pooled[i + k][j + l] * C2Filters[filter][channel][k][l];
                                    }
                                    else
                                    {
                                        Sum += C1Image2Pooled[i + k][j + l] * C2Filters[filter][channel][k][l];
                                    }
                                }
                            }
                        }

                        if (filter == 0)
                        {
                            C2Image1[i][j] = ReLU(Sum + C2F1Bias);
                        }

                        if (filter == 1)
                        {
                            C2Image2[i][j] = ReLU(Sum + C2F2Bias);
                        }

                        if (filter == 2)
                        {
                            C2Image3[i][j] = ReLU(Sum + C2F3Bias);
                        }

                        if (filter == 3)
                        {
                            C2Image4[i][j] = ReLU(Sum + C2F4Bias);
                        }
                    }
                }
            }

            for (int i = 0; i < 30; i++)
            {
                for (int j = 0; j < 30; j++)
                {
                    int r = i * 2;
                    int c = j * 2;

                    float Max1 = C2Image1[r][c];
                    float Max2 = C2Image2[r][c];
                    float Max3 = C2Image3[r][c];
                    float Max4 = C2Image4[r][c];

                    for (int k = 0; k < 2; k++)
                    {
                        for (int l = 0; l < 2; l++)
                        {
                            if (C2Image1[r + k][c + l] > Max1)
                            {
                                Max1 = C2Image1[r + k][c + l];
                            }

                            if (C2Image2[r + k][c + l] > Max2)
                            {
                                Max2 = C2Image2[r + k][c + l];
                            }

                            if (C2Image3[r + k][c + l] > Max3)
                            {
                                Max3 = C2Image3[r + k][c + l];
                            }

                            if (C2Image4[r + k][c + l] > Max4)
                            {
                                Max4 = C2Image4[r + k][c + l];
                            }
                        }
                    }

                    C2Image1Pooled[i][j] = Max1;
                    C2Image2Pooled[i][j] = Max2;
                    C2Image3Pooled[i][j] = Max3;
                    C2Image4Pooled[i][j] = Max4;
                }
            }

            int FlatIndex = 0;

            for (int i = 0; i < 30; i++)
            {
                for (int j = 0; j < 30; j++)
                {
                    FlatPixelValues[FlatIndex++] = C2Image1Pooled[i][j];
                }
            }

            for (int i = 0; i < 30; i++)
            {
                for (int j = 0; j < 30; j++)
                {
                    FlatPixelValues[FlatIndex++] = C2Image2Pooled[i][j];
                }
            }

            for (int i = 0; i < 30; i++)
            {
                for (int j = 0; j < 30; j++)
                {
                    FlatPixelValues[FlatIndex++] = C2Image3Pooled[i][j];
                }
            }

            for (int i = 0; i < 30; i++)
            {
                for (int j = 0; j < 30; j++)
                {
                    FlatPixelValues[FlatIndex++] = C2Image4Pooled[i][j];
                }
            }

            for (int i = 0; i < 2; i++)
            {
                Output[i] = OutputBiases[i];

                for (int j = 0; j < 3600; j++)
                {
                    Output[i] += FlatPixelValues[j] * FlatToOutputWeights[i][j];
                }
            }

            vector<float> Probabilities = Softmax(Output);

            int Prediction = max_element(Probabilities.begin(), Probabilities.end()) - Probabilities.begin();

            int Label = TrainingLabels[img];

            if (Prediction == Label)
            {
                Correct++;
            }

            float Probability = max(Probabilities[Label], 1e-10f);

            TotalLoss -= log(Probability);

            for (int i = 0; i < 2; i++)
            {
                if (i == Label)
                {
                    OutputError[i] = Probabilities[i] - 1.0f;
                }
                else
                {
                    OutputError[i] = Probabilities[i];
                }
            }

            for (int j = 0; j < 3600; j++)
            {
                FlatError[j] = 0.0f;

                for (int i = 0; i < 2; i++)
                {
                    FlatError[j] += OutputError[i] * FlatToOutputWeights[i][j];
                }
            }

            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 3600; j++)
                {
                    FlatToOutputWeights[i][j] -= LearningRate * OutputError[i] * FlatPixelValues[j];
                }

                OutputBiases[i] -= LearningRate * OutputError[i];
            }

            for (int filter = 0; filter < 4; filter++)
            {
                for (int r = 0; r < 30; r++)
                {
                    for (int c = 0; c < 30; c++)
                    {
                        C2PooledErrors[filter][r * 30 + c] =
                            FlatError[filter * 900 + r * 30 + c];
                    }
                }
            }

            for (int r = 0; r < 60; r++)
            {
                for (int c = 0; c < 60; c++)
                {
                    C2Error1[r][c] = 0.0f;
                    C2Error2[r][c] = 0.0f;
                    C2Error3[r][c] = 0.0f;
                    C2Error4[r][c] = 0.0f;
                }
            }

            for (int filter = 0; filter < 4; filter++)
            {
                for (int pr = 0; pr < 30; pr++)
                {
                    for (int pc = 0; pc < 30; pc++)
                    {
                        int r = pr * 2;
                        int c = pc * 2;

                        float MaxValue;
                        int MaxRow;
                        int MaxCol;

                        if (filter == 0)
                        {
                            MaxValue = C2Image1[r][c];
                        }
                        else if (filter == 1)
                        {
                            MaxValue = C2Image2[r][c];
                        }
                        else if (filter == 2)
                        {
                            MaxValue = C2Image3[r][c];
                        }
                        else
                        {
                            MaxValue = C2Image4[r][c];
                        }

                        MaxRow = r;
                        MaxCol = c;

                        for (int k = 0; k < 2; k++)
                        {
                            for (int l = 0; l < 2; l++)
                            {
                                float Value;

                                if (filter == 0)
                                {
                                    Value = C2Image1[r + k][c + l];
                                }
                                else if (filter == 1)
                                {
                                    Value = C2Image2[r + k][c + l];
                                }
                                else if (filter == 2)
                                {
                                    Value = C2Image3[r + k][c + l];
                                }
                                else
                                {
                                    Value = C2Image4[r + k][c + l];
                                }

                                if (Value > MaxValue)
                                {
                                    MaxValue = Value;
                                    MaxRow = r + k;
                                    MaxCol = c + l;
                                }
                            }
                        }

                        float Error = C2PooledErrors[filter][pr * 30 + pc];

                        if (filter == 0)
                        {
                            C2Error1[MaxRow][MaxCol] = Error;
                        }
                        else if (filter == 1)
                        {
                            C2Error2[MaxRow][MaxCol] = Error;
                        }
                        else if (filter == 2)
                        {
                            C2Error3[MaxRow][MaxCol] = Error;
                        }
                        else
                        {
                            C2Error4[MaxRow][MaxCol] = Error;
                        }
                    }
                }
            }

            for (int r = 0; r < 60; r++)
            {
                for (int c = 0; c < 60; c++)
                {
                    if (C2Image1[r][c] <= 0.0f)
                    {
                        C2Error1[r][c] = 0.0f;
                    }

                    if (C2Image2[r][c] <= 0.0f)
                    {
                        C2Error2[r][c] = 0.0f;
                    }

                    if (C2Image3[r][c] <= 0.0f)
                    {
                        C2Error3[r][c] = 0.0f;
                    }

                    if (C2Image4[r][c] <= 0.0f)
                    {
                        C2Error4[r][c] = 0.0f;
                    }
                }
            }

            for (int i = 0; i < 18; i++)
            {
                C2DLoss1[i] = 0.0f;
                C2DLoss2[i] = 0.0f;
                C2DLoss3[i] = 0.0f;
                C2DLoss4[i] = 0.0f;
            }

            for (int filter = 0; filter < 4; filter++)
            {
                for (int k = 0; k < 3; k++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        float LossChannel1 = 0.0f;
                        float LossChannel2 = 0.0f;

                        for (int r = 0; r < 60; r++)
                        {
                            for (int c = 0; c < 60; c++)
                            {
                                float Error;

                                if (filter == 0)
                                {
                                    Error = C2Error1[r][c];
                                }
                                else if (filter == 1)
                                {
                                    Error = C2Error2[r][c];
                                }
                                else if (filter == 2)
                                {
                                    Error = C2Error3[r][c];
                                }
                                else
                                {
                                    Error = C2Error4[r][c];
                                }

                                LossChannel1 +=
                                    Error *
                                    C1Image1Pooled[r + k][c + l];

                                LossChannel2 +=
                                    Error *
                                    C1Image2Pooled[r + k][c + l];
                            }
                        }

                        if (filter == 0)
                        {
                            C2DLoss1[k * 3 + l] = LossChannel1;
                            C2DLoss1[9 + k * 3 + l] = LossChannel2;
                        }
                        else if (filter == 1)
                        {
                            C2DLoss2[k * 3 + l] = LossChannel1;
                            C2DLoss2[9 + k * 3 + l] = LossChannel2;
                        }
                        else if (filter == 2)
                        {
                            C2DLoss3[k * 3 + l] = LossChannel1;
                            C2DLoss3[9 + k * 3 + l] = LossChannel2;
                        }
                        else
                        {
                            C2DLoss4[k * 3 + l] = LossChannel1;
                            C2DLoss4[9 + k * 3 + l] = LossChannel2;
                        }
                    }
                }
            }

            float C2BiasLoss1 = 0.0f;
            float C2BiasLoss2 = 0.0f;
            float C2BiasLoss3 = 0.0f;
            float C2BiasLoss4 = 0.0f;

            for (int r = 0; r < 60; r++)
            {
                for (int c = 0; c < 60; c++)
                {
                    C2BiasLoss1 += C2Error1[r][c];
                    C2BiasLoss2 += C2Error2[r][c];
                    C2BiasLoss3 += C2Error3[r][c];
                    C2BiasLoss4 += C2Error4[r][c];
                }
            }

            for (int r = 0; r < 62; r++)
            {
                for (int c = 0; c < 62; c++)
                {
                    C1PooledError1[r][c] = 0.0f;
                    C1PooledError2[r][c] = 0.0f;
                }
            }

            for (int r = 0; r < 60; r++)
            {
                for (int c = 0; c < 60; c++)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        for (int l = 0; l < 3; l++)
                        {
                            C1PooledError1[r + k][c + l] +=
                                C2Error1[r][c] *
                                C2Filters[0][0][k][l];

                            C1PooledError2[r + k][c + l] +=
                                C2Error1[r][c] *
                                C2Filters[0][1][k][l];

                            C1PooledError1[r + k][c + l] +=
                                C2Error2[r][c] *
                                C2Filters[1][0][k][l];

                            C1PooledError2[r + k][c + l] +=
                                C2Error2[r][c] *
                                C2Filters[1][1][k][l];

                            C1PooledError1[r + k][c + l] +=
                                C2Error3[r][c] *
                                C2Filters[2][0][k][l];

                            C1PooledError2[r + k][c + l] +=
                                C2Error3[r][c] *
                                C2Filters[2][1][k][l];

                            C1PooledError1[r + k][c + l] +=
                                C2Error4[r][c] *
                                C2Filters[3][0][k][l];

                            C1PooledError2[r + k][c + l] +=
                                C2Error4[r][c] *
                                C2Filters[3][1][k][l];
                        }
                    }
                }
            }

            for (int r = 0; r < 124; r++)
            {
                for (int c = 0; c < 124; c++)
                {
                    C1Error1[r][c] = 0.0f;
                    C1Error2[r][c] = 0.0f;
                }
            }

            for (int pr = 0; pr < 62; pr++)
            {
                for (int pc = 0; pc < 62; pc++)
                {
                    int r = pr * 2;
                    int c = pc * 2;

                    float Max1 = C1Image1[r][c];
                    int MaxRow1 = r;
                    int MaxCol1 = c;

                    float Max2 = C1Image2[r][c];
                    int MaxRow2 = r;
                    int MaxCol2 = c;

                    for (int k = 0; k < 2; k++)
                    {
                        for (int l = 0; l < 2; l++)
                        {
                            if (C1Image1[r + k][c + l] > Max1)
                            {
                                Max1 = C1Image1[r + k][c + l];
                                MaxRow1 = r + k;
                                MaxCol1 = c + l;
                            }

                            if (C1Image2[r + k][c + l] > Max2)
                            {
                                Max2 = C1Image2[r + k][c + l];
                                MaxRow2 = r + k;
                                MaxCol2 = c + l;
                            }
                        }
                    }

                    C1Error1[MaxRow1][MaxCol1] = C1PooledError1[pr][pc];
                    C1Error2[MaxRow2][MaxCol2] = C1PooledError2[pr][pc];
                }
            }

            for (int r = 0; r < 124; r++)
            {
                for (int c = 0; c < 124; c++)
                {
                    if (C1Image1[r][c] <= 0.0f)
                    {
                        C1Error1[r][c] = 0.0f;
                    }

                    if (C1Image2[r][c] <= 0.0f)
                    {
                        C1Error2[r][c] = 0.0f;
                    }
                }
            }

            for (int k = 0; k < 5; k++)
            {
                for (int l = 0; l < 5; l++)
                {
                    C1DLoss1[k * 5 + l] = 0.0f;
                    C1DLoss2[k * 5 + l] = 0.0f;

                    for (int r = 0; r < 124; r++)
                    {
                        for (int c = 0; c < 124; c++)
                        {
                            C1DLoss1[k * 5 + l] +=
                                C1Error1[r][c] *
                                TrainingImagesData[img][(r + k) * 128 + (c + l)];

                            C1DLoss2[k * 5 + l] +=
                                C1Error2[r][c] *
                                TrainingImagesData[img][(r + k) * 128 + (c + l)];
                        }
                    }
                }
            }

            float C1BiasLoss1 = 0.0f;
            float C1BiasLoss2 = 0.0f;

            for (int r = 0; r < 124; r++)
            {
                for (int c = 0; c < 124; c++)
                {
                    C1BiasLoss1 += C1Error1[r][c];
                    C1BiasLoss2 += C1Error2[r][c];
                }
            }

            for (int filter = 0; filter < 4; filter++)
            {
                for (int k = 0; k < 3; k++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        if (filter == 0)
                        {
                            C2Filters[0][0][k][l] -=
                                LearningRate *
                                C2DLoss1[k * 3 + l];

                            C2Filters[0][1][k][l] -=
                                LearningRate *
                                C2DLoss1[9 + k * 3 + l];
                        }

                        if (filter == 1)
                        {
                            C2Filters[1][0][k][l] -=
                                LearningRate *
                                C2DLoss2[k * 3 + l];

                            C2Filters[1][1][k][l] -=
                                LearningRate *
                                C2DLoss2[9 + k * 3 + l];
                        }

                        if (filter == 2)
                        {
                            C2Filters[2][0][k][l] -=
                                LearningRate *
                                C2DLoss3[k * 3 + l];

                            C2Filters[2][1][k][l] -=
                                LearningRate *
                                C2DLoss3[9 + k * 3 + l];
                        }

                        if (filter == 3)
                        {
                            C2Filters[3][0][k][l] -=
                                LearningRate *
                                C2DLoss4[k * 3 + l];

                            C2Filters[3][1][k][l] -=
                                LearningRate *
                                C2DLoss4[9 + k * 3 + l];
                        }
                    }
                }
            }

            C2F1Bias -= LearningRate * C2BiasLoss1;
            C2F2Bias -= LearningRate * C2BiasLoss2;
            C2F3Bias -= LearningRate * C2BiasLoss3;
            C2F4Bias -= LearningRate * C2BiasLoss4;

            for (int k = 0; k < 5; k++)
            {
                for (int l = 0; l < 5; l++)
                {
                    C1Filter1[k][l] -=
                        LearningRate *
                        C1DLoss1[k * 5 + l];

                    C1Filter2[k][l] -=
                        LearningRate *
                        C1DLoss2[k * 5 + l];
                }
            }

            C1F1Bias -= LearningRate * C1BiasLoss1;
            C1F2Bias -= LearningRate * C1BiasLoss2;
        }

        float AverageLoss = TotalLoss / images;
        float Accuracy = (float)Correct / images * 100.0f;

        cout << "Loss: " << AverageLoss << endl;
        cout << "Accuracy: " << Accuracy << "%" << endl;

        SaveWeights();
    }
}

int main()
{
    TrainCNN();

    return 0;
}