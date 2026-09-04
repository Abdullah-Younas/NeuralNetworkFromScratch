#pragma once

void InitializeCNN_CUDA(
    const float* C1Filters,
    const float* C2Filters,
    const float* DenseWeights,
    const float* OutputBiases,
    float C1F1Bias,
    float C1F2Bias,
    float C2F1Bias,
    float C2F2Bias,
    float C2F3Bias,
    float C2F4Bias
);

void TrainImageCUDA(
    const float* input,
    int label,
    float DenseLearningRate,
    float ConvLearningRate,
    float* Probabilities,
    float& Loss,
    int& Prediction
);

void DownloadCNNWeights(
    float* C1Filters,
    float* C2Filters,
    float* DenseWeights,
    float* OutputBiases,
    float& C1F1Bias,
    float& C1F2Bias,
    float& C2F1Bias,
    float& C2F2Bias,
    float& C2F3Bias,
    float& C2F4Bias
);

void FreeCNN_CUDA();