#include "main.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void MatMulKernel(double* A, double* B, double* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.0;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void RunMatMul(std::vector<double>* inputs,
    std::vector<std::vector<double>>& weights,
    std::vector<double>* hidden,
    int M, int N, int K) {

    // weights is stored as [N][K] (outputSize x inputSize), e.g.
    // weightsInputHidden[hidden][input] or weightsHiddenOutput[output][hidden]
    // The kernel needs B laid out as K x N (row-major), so we transpose while flattening.

    size_t total_elements_weights = (size_t)K * N;
    size_t size_bytes = total_elements_weights * sizeof(double);

    std::vector<double> flatWeights(total_elements_weights);

    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            flatWeights[k * N + n] = weights[n][k];   // transpose here
        }
    }

    double* d_A, * d_B, * d_C;
    size_t sizeA = (size_t)M * K * sizeof(double);
    size_t sizeC = (size_t)M * N * sizeof(double);

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, size_bytes);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, inputs->data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, flatWeights.data(), size_bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    MatMulKernel << <gridDim, blockDim >> > (d_A, d_B, d_C, M, N, K);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(hidden->data(), d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
