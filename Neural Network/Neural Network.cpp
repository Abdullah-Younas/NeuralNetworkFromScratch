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
#include "cnn_cuda.cuh"

using namespace std;

const int OutputLayers = 2;

float* TrainingImagesData;
int* TrainingLabels;

int imageIndex = 0;

float C1Filters[2 * 5 * 5];
float C2Filters[4 * 2 * 3 * 3];

float C1Images[2 * 124 * 124];
float C1ImagesPooled[2 * 62 * 62];

float C2Images[4 * 60 * 60];
float C2ImagesPooled[4 * 30 * 30];

float FlatPixelValues[3600];
float FlatToOutputWeights[2 * 3600];

float OutputBiases[2];

float Output[2];
float OutputError[2];
float FlatError[3600];

float C2PooledErrors[4 * 900];
float C2Error[4 * 60 * 60];

float C1PooledError[2 * 62 * 62];
float C1Error[2 * 124 * 124];

float C2DLoss[4 * 18];
float C1DLoss[2 * 25];

float C1F1Bias = 0.0f;
float C1F2Bias = 0.0f;

float C2F1Bias = 0.0f;
float C2F2Bias = 0.0f;
float C2F3Bias = 0.0f;
float C2F4Bias = 0.0f;

int width, height, channels;

int tar_w = 128;
int tar_h = 128;

void LoadTrainingImagesCats()
{
    for (const auto& file : filesystem::directory_iterator("Cats"))
    {
        string filename = file.path().string();

        unsigned char* img = stbi_load(filename.c_str(), &width, &height, &channels, 1);

        if (img)
        {
            unsigned char* output = (unsigned char*)malloc(tar_w * tar_h);

            stbir_resize_uint8_linear(img, width, height, 0, output, tar_w, tar_h, 0, STBIR_1CHANNEL);

            for (int i = 0; i < 16384; i++)
            {
                TrainingImagesData[imageIndex * 16384 + i] = output[i] / 255.0f;
            }

            TrainingLabels[imageIndex] = 0;
            imageIndex++;

            if (imageIndex % 1000 == 0)
            {
                cout << "Images loaded: " << imageIndex << endl;
            }

            free(output);
            stbi_image_free(img);
        }
    }
}

void LoadTrainingImagesDogs()
{
    for (const auto& file : filesystem::directory_iterator("Dogs"))
    {
        string filename = file.path().string();

        unsigned char* img = stbi_load(filename.c_str(), &width, &height, &channels, 1);

        if (img)
        {
            unsigned char* output = (unsigned char*)malloc(tar_w * tar_h);

            stbir_resize_uint8_linear(img, width, height, 0, output, tar_w, tar_h, 0, STBIR_1CHANNEL);

            for (int i = 0; i < 16384; i++)
            {
                TrainingImagesData[imageIndex * 16384 + i] = output[i] / 255.0f;
            }

            TrainingLabels[imageIndex] = 1;
            imageIndex++;

            if (imageIndex % 1000 == 0)
            {
                cout << "Images loaded: " << imageIndex << endl;
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
        for (int filter = 0; filter < 2; filter++)
        {
            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    inFile >> C1Filters[filter * 25 + i * 5 + j];
                }
            }
        }

        inFile.close();
        return;
    }

    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<float> dis(-0.1f, 0.1f);

    ofstream outFile("C1Weights.txt");

    for (int filter = 0; filter < 2; filter++)
    {
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                C1Filters[filter * 25 + i * 5 + j] = dis(gen);

                outFile << C1Filters[filter * 25 + i * 5 + j] << '\n';
            }
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
                        inFile >> C2Filters[filter * 18 + channel * 9 + i * 3 + j];
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
                    C2Filters[filter * 18 + channel * 9 + i * 3 + j] = dis(gen);

                    outFile << C2Filters[filter * 18 + channel * 9 + i * 3 + j] << '\n';
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
                inFile >> FlatToOutputWeights[i * 3600 + j];
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
            FlatToOutputWeights[i * 3600 + j] = dis(gen);

            outFile << FlatToOutputWeights[i * 3600 + j] << '\n';
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
    float MaxLogit = *max_element(Logits.begin(), Logits.end());

    float TotalSum = 0.0f;

    for (int i = 0; i < Logits.size(); i++)
    {
        Logits[i] = exp(Logits[i] - MaxLogit);

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
            DenseOut << FlatToOutputWeights[i * 3600 + j] << '\n';
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

    for (int filter = 0; filter < 2; filter++)
    {
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                C1Out << C1Filters[filter * 25 + i * 5 + j] << '\n';
            }
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
                    C2Out << C2Filters[filter * 18 + channel * 9 + i * 3 + j] << '\n';
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
    int images = 19932;

    TrainingImagesData = new float[images * 128 * 128];
    TrainingLabels = new int[images];

    LoadTrainingImagesCats();
    LoadTrainingImagesDogs();

    cout << "Total Images loaded: " << imageIndex << endl;
    cout << "Total Labels loaded: " << imageIndex << endl;
    cout << "Image size: 16384 pixels" << endl;

    LoadRandomWeightsIntoC1Filters();
    LoadRandomWeightsIntoC2Filters();
    LoadRandomWeightsIntoFlatToOutputWeights();

    LoadRandomBiasesIntoOutputLayer();
    LoadC1C2Biases();

    InitializeCNN_CUDA(
        C1Filters,
        C2Filters,
        FlatToOutputWeights,
        OutputBiases,
        C1F1Bias,
        C1F2Bias,
        C2F1Bias,
        C2F2Bias,
        C2F3Bias,
        C2F4Bias
    );

    float Probabilities[2];

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float LearningRate = 0.0f;
        float TotalLoss = 0.0f;
        int Correct = 0;

        cout << "Starting Epoch "
            << epoch + 1
            << "/"
            << epochs
            << endl;

        for (int img = 0; img < images; img++)
        {
            float Loss;
            int Prediction;

            TrainImageCUDA(
                &TrainingImagesData[img * 16384],
                TrainingLabels[img],
                LearningRate,
                Probabilities,
                Loss,
                Prediction
            );

            TotalLoss += Loss;

            if (Prediction == TrainingLabels[img])
                Correct++;

            if ((img + 1) % 1000 == 0)
            {
                cout << "Image: "
                    << img + 1
                    << "/"
                    << images
                    << endl;
            }
        }

        DownloadCNNWeights(
            C1Filters,
            C2Filters,
            FlatToOutputWeights,
            OutputBiases,
            C1F1Bias,
            C1F2Bias,
            C2F1Bias,
            C2F2Bias,
            C2F3Bias,
            C2F4Bias
        );

        float AverageLoss =
            TotalLoss / images;

        float Accuracy =
            (float)Correct /
            images *
            100.0f;

        cout << "Epoch "
            << epoch + 1
            << " Loss: "
            << AverageLoss
            << endl;

        cout << "Epoch "
            << epoch + 1
            << " Accuracy: "
            << Accuracy
            << "%"
            << endl;

        SaveWeights();
    }

    FreeCNN_CUDA();

    delete[] TrainingImagesData;
    delete[] TrainingLabels;
}

int main()
{
    TrainCNN();

    return 0;
}