#include <iostream>
#include <random>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <vector>
#include <cstdint> 

using namespace std;

vector<vector<vector<float>>> TrainingImagesData;

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

vector<float> Output(10);

void LoadTrainingDataMNIST(int imgsToLoad)
{
    ifstream file("train-images-idx3-ubyte", ios::binary);

    if (!file.is_open())
    {
        cout << "Failed to open MNIST file!" << endl;
        return;
    }

    cout << "MNIST file opened!" << endl;

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

void LoadRandomWeightsIntoC1Filters() {
    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<float> dis(-1.0f, nextafter(1.0f, 2.0f));

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            C1Filter1[i][j] = dis(gen);
            C1Filter2[i][j] = dis(gen);
        }
    }
}

void LoadRandomWeightsIntoC2Filters() {
    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<float> dis(-1.0f, nextafter(1.0f, 2.0f));

    for (int filters = 0; filters < 4; filters++) {
        for (int channel = 0; channel < 2; channel++) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    C2Filters[filters][channel][i][j] = dis(gen);
                }
            }
        }
    }
}

void LoadRandomWeightsIntoFlatToOutputWeights() {
    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<float> dis(-1.0f, nextafter(1.0f, 2.0f));
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 100; j++) {
            FlatToOutputWeights[i][j] = dis(gen);
        }
    }
}

void LoadRandomBiasesIntoOutputLayer() {
    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<float> dis(-1.0f, nextafter(1.0f, 2.0f));
    for (int i = 0; i < 10; i++) {
        OutputBiases[i] = dis(gen);
    }
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

int main()
{
    //Loading the Images 1 means 1 images will be loaded
    LoadTrainingDataMNIST(1);

    //Initializing the C1 kernels with random weights
    LoadRandomWeightsIntoC1Filters();

    //Initializing the C2 kernels with random weights
    LoadRandomWeightsIntoC2Filters();

    //Initializing the Flat to Output layer with random weights
    LoadRandomWeightsIntoFlatToOutputWeights();

    //Initializing Output layer with random biases
    LoadRandomBiasesIntoOutputLayer();

    //Convolution 1
    int C1rows = 28;
    int C1cols = 28;
    int C1kernelSize = 5;
    float C1F1Bias = 0.155f;
    float C1F2Bias = 0.255f;

    //Pooling
    int PoolSize = 2;
    int PoolStride = 2;

    //Convolution 2
    int C2rows = 12;
    int C2cols = 12;
    int C2KernelSize = 3;
    float C2F1Bias = -0.255f;
    float C2F2Bias = -0.155f;
    float C2F3Bias = 0.125f;
    float C2F4Bias = 0.225f;

    //Flattening Pooled images into a single vector/array
    int FAIndex = 0;

    //Fully Connected layer Flat - Output
    int OutputLayers = 10;

    //Convolution 1 Kernel Processed Images
    for (int i = 0; i <= C1rows - C1kernelSize; i++) {
        for (int j = 0; j <= C1cols - C1kernelSize; j++) {

            float sum1 = 0.0f;
            float sum2 = 0.0f;

            for (int k = 0; k < C1kernelSize; k++) {
                for (int l = 0; l < C1kernelSize; l++) {

                    sum1 += TrainingImagesData[0][i + k][j + l] * C1Filter1[k][l];

                    sum2 += TrainingImagesData[0][i + k][j + l] * C1Filter2[k][l];

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

    for (int i = 0; i < OutputLayers; i++) {
        cout << "Probabilities[" << i << "]: " << Probabilities[i] << endl;
    }

    return 0;
}

