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
                cout << "Images loaded: " << imageIndex << endl;

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
                cout << "Images loaded: " << imageIndex << endl;

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
                    inFile >> C1Filters[filter * 25 + i * 5 + j];
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
                        inFile >> C2Filters[filter * 18 + channel * 9 + i * 3 + j];
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
                inFile >> FlatToOutputWeights[i * 3600 + j];
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
            inFile >> OutputBiases[i];

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
    return value > 0.0f ? value : 0.0f;
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
        Logits[i] /= TotalSum;

    return Logits;
}

void SaveWeights()
{
    ofstream DenseOut("DenseWeights.txt");

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3600; j++)
            DenseOut << FlatToOutputWeights[i * 3600 + j] << '\n';
    }

    DenseOut.close();

    ofstream OutputBiasOut("OutputBiases.txt");

    for (int i = 0; i < 2; i++)
        OutputBiasOut << OutputBiases[i] << '\n';

    OutputBiasOut.close();

    ofstream C1Out("C1Weights.txt");

    for (int filter = 0; filter < 2; filter++)
    {
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
                C1Out << C1Filters[filter * 25 + i * 5 + j] << '\n';
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
                    C2Out << C2Filters[filter * 18 + channel * 9 + i * 3 + j] << '\n';
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

void PredictImageCPU(const float* input, int& prediction, float& catProbability, float& dogProbability)
{
    for (int filter = 0; filter < 2; filter++)
    {
        float bias = filter == 0 ? C1F1Bias : C1F2Bias;

        for (int r = 0; r < 124; r++)
        {
            for (int c = 0; c < 124; c++)
            {
                float sum = bias;

                for (int i = 0; i < 5; i++)
                {
                    for (int j = 0; j < 5; j++)
                    {
                        sum += input[(r + i) * 128 + (c + j)] *
                            C1Filters[filter * 25 + i * 5 + j];
                    }
                }

                C1Images[filter * 15376 + r * 124 + c] = ReLU(sum);
            }
        }
    }

    for (int channel = 0; channel < 2; channel++)
    {
        for (int r = 0; r < 62; r++)
        {
            for (int c = 0; c < 62; c++)
            {
                int base = channel * 15376;

                int i0 = base + (r * 2) * 124 + c * 2;
                int i1 = i0 + 1;
                int i2 = i0 + 124;
                int i3 = i2 + 1;

                float maxValue = C1Images[i0];

                if (C1Images[i1] > maxValue)
                    maxValue = C1Images[i1];

                if (C1Images[i2] > maxValue)
                    maxValue = C1Images[i2];

                if (C1Images[i3] > maxValue)
                    maxValue = C1Images[i3];

                C1ImagesPooled[channel * 3844 + r * 62 + c] = maxValue;
            }
        }
    }

    for (int filter = 0; filter < 4; filter++)
    {
        float bias;

        if (filter == 0)
            bias = C2F1Bias;
        else if (filter == 1)
            bias = C2F2Bias;
        else if (filter == 2)
            bias = C2F3Bias;
        else
            bias = C2F4Bias;

        for (int r = 0; r < 60; r++)
        {
            for (int c = 0; c < 60; c++)
            {
                float sum = bias;

                for (int channel = 0; channel < 2; channel++)
                {
                    for (int i = 0; i < 3; i++)
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            sum +=
                                C1ImagesPooled[channel * 3844 + (r + i) * 62 + (c + j)] *
                                C2Filters[filter * 18 + channel * 9 + i * 3 + j];
                        }
                    }
                }

                C2Images[filter * 3600 + r * 60 + c] = ReLU(sum);
            }
        }
    }

    for (int channel = 0; channel < 4; channel++)
    {
        for (int r = 0; r < 30; r++)
        {
            for (int c = 0; c < 30; c++)
            {
                int base = channel * 3600;

                int i0 = base + (r * 2) * 60 + c * 2;
                int i1 = i0 + 1;
                int i2 = i0 + 60;
                int i3 = i2 + 1;

                float maxValue = C2Images[i0];

                if (C2Images[i1] > maxValue)
                    maxValue = C2Images[i1];

                if (C2Images[i2] > maxValue)
                    maxValue = C2Images[i2];

                if (C2Images[i3] > maxValue)
                    maxValue = C2Images[i3];

                C2ImagesPooled[channel * 900 + r * 30 + c] = maxValue;
            }
        }
    }

    for (int i = 0; i < 3600; i++)
        FlatPixelValues[i] = C2ImagesPooled[i];

    float Logits[2];

    for (int output = 0; output < 2; output++)
    {
        Logits[output] = OutputBiases[output];

        for (int i = 0; i < 3600; i++)
        {
            Logits[output] +=
                FlatToOutputWeights[output * 3600 + i] *
                FlatPixelValues[i];
        }
    }

    vector<float> Probabilities =
        Softmax({ Logits[0], Logits[1] });

    catProbability = Probabilities[0];
    dogProbability = Probabilities[1];

    prediction =
        catProbability > dogProbability ? 0 : 1;
}

bool LoadTestImage(const string& filename, vector<float>& image)
{
    int testWidth;
    int testHeight;
    int testChannels;

    unsigned char* img =
        stbi_load(
            filename.c_str(),
            &testWidth,
            &testHeight,
            &testChannels,
            1
        );

    if (!img)
        return false;

    unsigned char* output =
        (unsigned char*)malloc(tar_w * tar_h);

    stbir_resize_uint8_linear(
        img,
        testWidth,
        testHeight,
        0,
        output,
        tar_w,
        tar_h,
        0,
        STBIR_1CHANNEL
    );

    image.resize(16384);

    for (int i = 0; i < 16384; i++)
        image[i] = output[i] / 255.0f;

    free(output);
    stbi_image_free(img);

    return true;
}

bool WeightFilesExist()
{
    ifstream c1("C1Weights.txt");
    ifstream c2("C2Weights.txt");
    ifstream dense("DenseWeights.txt");
    ifstream outputBias("OutputBiases.txt");
    ifstream c1c2("C1C2Biases.txt");

    return c1.good() &&
        c2.good() &&
        dense.good() &&
        outputBias.good() &&
        c1c2.good();
}

void TestCNN()
{
    if (!WeightFilesExist())
    {
        cout << "Trained weight files not found." << endl;
        cout << "Train the CNN first." << endl;
        return;
    }

    LoadRandomWeightsIntoC1Filters();
    LoadRandomWeightsIntoC2Filters();
    LoadRandomWeightsIntoFlatToOutputWeights();
    LoadRandomBiasesIntoOutputLayer();
    LoadC1C2Biases();

    vector<pair<string, int>> testImages;

    if (!filesystem::exists("test/cats") ||
        !filesystem::exists("test/dogs"))
    {
        cout << "Test folders not found." << endl;
        cout << "Expected: test/cat and test/dog" << endl;
        return;
    }

    for (const auto& file : filesystem::directory_iterator("test/cats"))
    {
        if (file.is_regular_file())
            testImages.push_back({ file.path().string(), 0 });
    }

    for (const auto& file : filesystem::directory_iterator("test/dogs"))
    {
        if (file.is_regular_file())
            testImages.push_back({ file.path().string(), 1 });
    }

    sort(
        testImages.begin(),
        testImages.end(),
        [](const pair<string, int>& a, const pair<string, int>& b)
        {
            return a.first < b.first;
        }
    );

    int correct = 0;
    int incorrect = 0;
    int failed = 0;

    cout << endl;
    cout << "Starting CPU Prediction Mode" << endl;
    cout << "Total test images: " << testImages.size() << endl;
    cout << endl;

    vector<float> image;

    for (int i = 0; i < testImages.size(); i++)
    {
        string filename = testImages[i].first;
        int actualLabel = testImages[i].second;

        if (!LoadTestImage(filename, image))
        {
            failed++;

            cout << "[" << i + 1 << "/" << testImages.size() << "] "
                << filesystem::path(filename).filename().string()
                << " -> FAILED TO LOAD"
                << endl;

            continue;
        }

        int prediction;
        float catProbability;
        float dogProbability;

        PredictImageCPU(
            image.data(),
            prediction,
            catProbability,
            dogProbability
        );

        string actual =
            actualLabel == 0 ? "Cat" : "Dog";

        string predicted =
            prediction == 0 ? "Cat" : "Dog";

        bool right =
            prediction == actualLabel;

        if (right)
            correct++;
        else
            incorrect++;

        cout << "[" << i + 1 << "/" << testImages.size() << "] "
            << filesystem::path(filename).filename().string()
            << " -> "
            << predicted
            << " | Cat: "
            << catProbability * 100.0f
            << "% | Dog: "
            << dogProbability * 100.0f
            << "% | "
            << (right ? "RIGHT" : "WRONG")
            << " | Actual: "
            << actual
            << endl;
    }

    int tested = correct + incorrect;

    float accuracy =
        tested > 0
        ? (float)correct / tested * 100.0f
        : 0.0f;

    cout << endl;
    cout << "========================================" << endl;
    cout << "TEST RESULTS" << endl;
    cout << "========================================" << endl;
    cout << "Total images: " << testImages.size() << endl;
    cout << "Tested: " << tested << endl;
    cout << "Correct: " << correct << endl;
    cout << "Incorrect: " << incorrect << endl;
    cout << "Failed to load: " << failed << endl;
    cout << "Accuracy: " << accuracy << "%" << endl;
    cout << "========================================" << endl;
}

void TrainCNN()
{
    int epochs = 20;
    int images = 19932;

    TrainingImagesData = new float[images * 128 * 128];
    TrainingLabels = new int[images];

    imageIndex = 0;

    LoadTrainingImagesCats();
    LoadTrainingImagesDogs();

    cout << "Total Images loaded: " << imageIndex << endl;
    cout << "Total Labels loaded: " << imageIndex << endl;
    cout << "Image size: 16384 pixels" << endl;

    if (imageIndex != images)
    {
        images = imageIndex;
    }

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

    vector<int> indices(images);

    for (int i = 0; i < images; i++)
        indices[i] = i;

    random_device rd;
    mt19937 gen(rd());

    float Probabilities[2];

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        shuffle(indices.begin(), indices.end(), gen);

        float DenseLearningRate = 0.01f;
        float ConvLearningRate = 0.01f;
        float TotalLoss = 0.0f;

        int Correct = 0;
        int CatTotal = 0;
        int DogTotal = 0;
        int CatCorrect = 0;
        int DogCorrect = 0;

        cout << endl;
        cout << "========================================" << endl;
        cout << "Starting Epoch "
            << epoch + 1
            << "/"
            << epochs
            << endl;
        cout << "========================================" << endl;

        for (int img = 0; img < images; img++)
        {
            int index = indices[img];

            float Loss;
            int Prediction;

            TrainImageCUDA(
                &TrainingImagesData[index * 16384],
                TrainingLabels[index],
                DenseLearningRate,
                ConvLearningRate,
                Probabilities,
                Loss,
                Prediction
            );

            TotalLoss += Loss;

            int actualLabel = TrainingLabels[index];

            if (actualLabel == 0)
            {
                CatTotal++;

                if (Prediction == 0)
                    CatCorrect++;
            }
            else
            {
                DogTotal++;

                if (Prediction == 1)
                    DogCorrect++;
            }

            if (Prediction == actualLabel)
                Correct++;

            if ((img + 1) % 1000 == 0)
            {
                float CurrentAccuracy =
                    (float)Correct /
                    (img + 1) *
                    100.0f;

                cout << "Image: "
                    << img + 1
                    << "/"
                    << images
                    << " | Accuracy: "
                    << CurrentAccuracy
                    << "%"
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

        float CatAccuracy =
            CatTotal > 0
            ? (float)CatCorrect / CatTotal * 100.0f
            : 0.0f;

        float DogAccuracy =
            DogTotal > 0
            ? (float)DogCorrect / DogTotal * 100.0f
            : 0.0f;

        cout << endl;
        cout << "Epoch "
            << epoch + 1
            << " Results"
            << endl;

        cout << "Loss: "
            << AverageLoss
            << endl;

        cout << "Overall Accuracy: "
            << Accuracy
            << "%"
            << endl;

        cout << "Cat Accuracy: "
            << CatAccuracy
            << "% ("
            << CatCorrect
            << "/"
            << CatTotal
            << ")"
            << endl;

        cout << "Dog Accuracy: "
            << DogAccuracy
            << "% ("
            << DogCorrect
            << "/"
            << DogTotal
            << ")"
            << endl;

        cout << endl;

        float maxC1 = fabs(C1Filters[0]);
        float minC1 = fabs(C1Filters[0]);

        for (int i = 0; i < 50; i++)
        {
            float v = fabs(C1Filters[i]);

            if (v > maxC1) maxC1 = v;
            if (v < minC1) minC1 = v;
        }

        float maxC2 = fabs(C2Filters[0]);
        float minC2 = fabs(C2Filters[0]);

        for (int i = 0; i < 72; i++)
        {
            float v = fabs(C2Filters[i]);

            if (v > maxC2) maxC2 = v;
            if (v < minC2) minC2 = v;
        }

        float maxDense = fabs(FlatToOutputWeights[0]);
        float minDense = fabs(FlatToOutputWeights[0]);

        for (int i = 0; i < 7200; i++)
        {
            float v = fabs(FlatToOutputWeights[i]);

            if (v > maxDense) maxDense = v;
            if (v < minDense) minDense = v;
        }

        cout << "C1 Weights  | Min Abs: "
            << minC1
            << " Max Abs: "
            << maxC1
            << endl;

        cout << "C2 Weights  | Min Abs: "
            << minC2
            << " Max Abs: "
            << maxC2
            << endl;

        cout << "Dense Weights | Min Abs: "
            << minDense
            << " Max Abs: "
            << maxDense
            << endl;

        SaveWeights();
    }

    FreeCNN_CUDA();

    delete[] TrainingImagesData;
    delete[] TrainingLabels;
}

int main()
{
    int mode;

    cout << "0 = Training" << endl;
    cout << "1 = Prediction" << endl;
    cout << "Enter mode: ";

    cin >> mode;

    if (mode == 0)
    {
        TrainCNN();
    }
    else if (mode == 1)
    {
        TestCNN();
    }
    else
    {
        cout << "Invalid mode." << endl;
    }

    return 0;
}