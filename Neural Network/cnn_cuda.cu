#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "cnn_cuda.cuh"

static float* d_Input = nullptr;

static float* d_C1Filters = nullptr;
static float* d_C1Images = nullptr;
static float* d_C1ImagesPooled = nullptr;

static float* d_C2Filters = nullptr;
static float* d_C2Images = nullptr;
static float* d_C2ImagesPooled = nullptr;

static float* d_FlatPixelValues = nullptr;

static float* d_DenseWeights = nullptr;
static float* d_OutputBiases = nullptr;

static float* d_Output = nullptr;
static float* d_Probabilities = nullptr;
static float* d_OutputError = nullptr;
static float* d_FlatError = nullptr;

static float* d_C2Error = nullptr;
static float* d_C2DLoss = nullptr;
static float* d_C2BiasLoss = nullptr;

static float* d_C1PooledError = nullptr;
static float* d_C1Error = nullptr;
static float* d_C1DLoss = nullptr;
static float* d_C1BiasLoss = nullptr;

static float* d_C1Biases = nullptr;
static float* d_C2Biases = nullptr;

static void CheckCUDA(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        exit(1);
    }
}

__device__ float ReLUDevice(float x)
{
    return x > 0.0f ? x : 0.0f;
}

__global__ void C1ConvolutionKernel(
    const float* input,
    const float* filters,
    float* output,
    const float* biases)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int filter = blockIdx.z;

    if (x >= 124 || y >= 124 || filter >= 2)
        return;

    float sum = biases[filter];

    for (int k = 0; k < 5; k++)
    {
        for (int l = 0; l < 5; l++)
        {
            sum +=
                input[(y + k) * 128 + (x + l)] *
                filters[filter * 25 + k * 5 + l];
        }
    }

    output[
        filter * 15376 +
            y * 124 +
            x
    ] = ReLUDevice(sum);
}

__global__ void MaxPoolC1Kernel(
    const float* input,
    float* output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z;

    if (x >= 62 || y >= 62 || channel >= 2)
        return;

    int r = y * 2;
    int c = x * 2;
    int base = channel * 15376;

    float a = input[base + r * 124 + c];
    float b = input[base + r * 124 + c + 1];
    float d = input[base + (r + 1) * 124 + c];
    float e = input[base + (r + 1) * 124 + c + 1];

    float maxValue = a;

    if (b > maxValue)
        maxValue = b;

    if (d > maxValue)
        maxValue = d;

    if (e > maxValue)
        maxValue = e;

    output[
        channel * 3844 +
            y * 62 +
            x
    ] = maxValue;
}

__global__ void C2ConvolutionKernel(
    const float* input,
    const float* filters,
    float* output,
    const float* biases)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int filter = blockIdx.z;

    if (x >= 60 || y >= 60 || filter >= 4)
        return;

    float sum = biases[filter];

    for (int channel = 0; channel < 2; channel++)
    {
        for (int k = 0; k < 3; k++)
        {
            for (int l = 0; l < 3; l++)
            {
                sum +=
                    input[
                        channel * 3844 +
                            (y + k) * 62 +
                            (x + l)
                    ] *
                    filters[
                        filter * 18 +
                            channel * 9 +
                            k * 3 +
                            l
                    ];
            }
        }
    }

    output[
        filter * 3600 +
            y * 60 +
            x
    ] = ReLUDevice(sum);
}

__global__ void MaxPoolC2Kernel(
    const float* input,
    float* output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z;

    if (x >= 30 || y >= 30 || channel >= 4)
        return;

    int r = y * 2;
    int c = x * 2;
    int base = channel * 3600;

    float a = input[base + r * 60 + c];
    float b = input[base + r * 60 + c + 1];
    float d = input[base + (r + 1) * 60 + c];
    float e = input[base + (r + 1) * 60 + c + 1];

    float maxValue = a;

    if (b > maxValue)
        maxValue = b;

    if (d > maxValue)
        maxValue = d;

    if (e > maxValue)
        maxValue = e;

    output[
        channel * 900 +
            y * 30 +
            x
    ] = maxValue;
}

__global__ void FlattenKernel(
    const float* input,
    float* output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= 3600)
        return;

    output[index] = input[index];
}

__global__ void DenseForwardKernel(
    const float* input,
    const float* weights,
    const float* biases,
    float* output)
{
    int neuron = threadIdx.x;

    if (neuron >= 2)
        return;

    float sum = biases[neuron];

    for (int i = 0; i < 3600; i++)
    {
        sum +=
            input[i] *
            weights[neuron * 3600 + i];
    }

    output[neuron] = sum;
}

__global__ void SoftmaxKernel(
    const float* output,
    float* probabilities)
{
    if (threadIdx.x != 0)
        return;

    float maxValue = output[0];

    if (output[1] > maxValue)
        maxValue = output[1];

    float e0 = expf(output[0] - maxValue);
    float e1 = expf(output[1] - maxValue);

    float total = e0 + e1;

    probabilities[0] = e0 / total;
    probabilities[1] = e1 / total;
}

__global__ void OutputErrorKernel(
    const float* probabilities,
    float* outputError,
    int label)
{
    int i = threadIdx.x;

    if (i >= 2)
        return;

    outputError[i] =
        probabilities[i] -
        (i == label ? 1.0f : 0.0f);
}

__global__ void FlatErrorKernel(
    const float* outputError,
    const float* weights,
    float* flatError)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= 3600)
        return;

    flatError[j] =
        outputError[0] * weights[j] +
        outputError[1] * weights[3600 + j];
}

__global__ void DenseUpdateKernel(
    float* weights,
    float* biases,
    const float* outputError,
    const float* flatPixels,
    float learningRate)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= 7200)
        return;

    int neuron = index / 3600;
    int pixel = index % 3600;

    weights[index] -=
        learningRate *
        outputError[neuron] *
        flatPixels[pixel];

    if (pixel == 0)
    {
        biases[neuron] -=
            learningRate *
            outputError[neuron];
    }
}

__global__ void C2MaxPoolBackwardKernel(
    const float* input,
    const float* pooledError,
    float* error)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z;

    if (x >= 30 || y >= 30 || channel >= 4)
        return;

    int r = y * 2;
    int c = x * 2;

    int base = channel * 3600;

    int i0 = base + r * 60 + c;
    int i1 = base + r * 60 + c + 1;
    int i2 = base + (r + 1) * 60 + c;
    int i3 = base + (r + 1) * 60 + c + 1;

    float a = input[i0];
    float b = input[i1];
    float d = input[i2];
    float e = input[i3];

    float maxValue = a;
    int maxIndex = 0;

    if (b > maxValue)
    {
        maxValue = b;
        maxIndex = 1;
    }

    if (d > maxValue)
    {
        maxValue = d;
        maxIndex = 2;
    }

    if (e > maxValue)
    {
        maxValue = e;
        maxIndex = 3;
    }

    float errorValue =
        pooledError[
            channel * 900 +
                y * 30 +
                x
        ];

    error[i0] = 0.0f;
    error[i1] = 0.0f;
    error[i2] = 0.0f;
    error[i3] = 0.0f;

    if (maxIndex == 0)
        error[i0] = errorValue;
    else if (maxIndex == 1)
        error[i1] = errorValue;
    else if (maxIndex == 2)
        error[i2] = errorValue;
    else
        error[i3] = errorValue;
}

__global__ void C2ReluBackwardKernel(
    const float* images,
    float* error)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= 14400)
        return;

    if (images[index] <= 0.0f)
        error[index] = 0.0f;
}

__global__ void C2GradientKernel(
    const float* error,
    const float* input,
    float* gradients)
{
    int weightIndex = blockIdx.x;

    if (weightIndex >= 72)
        return;

    __shared__ float shared[256];

    int thread = threadIdx.x;

    int filter = weightIndex / 18;
    int remainder = weightIndex % 18;
    int channel = remainder / 9;
    int kernelIndex = remainder % 9;

    int k = kernelIndex / 3;
    int l = kernelIndex % 3;

    float sum = 0.0f;

    for (int index = thread; index < 3600; index += blockDim.x)
    {
        int r = index / 60;
        int c = index % 60;

        sum +=
            error[
                filter * 3600 +
                    index
            ] *
            input[
                channel * 3844 +
                    (r + k) * 62 +
                    (c + l)
            ];
    }

    shared[thread] = sum;

    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1)
    {
        if (thread < stride)
            shared[thread] += shared[thread + stride];

        __syncthreads();
    }

    if (thread == 0)
        gradients[weightIndex] = shared[0] / 3600.0f;
}

__global__ void C2BiasGradientKernel(
    const float* error,
    float* gradients)
{
    int filter = blockIdx.x;

    if (filter >= 4)
        return;

    __shared__ float shared[256];

    int thread = threadIdx.x;

    float sum = 0.0f;

    for (int index = thread; index < 3600; index += blockDim.x)
    {
        sum += error[
            filter * 3600 +
                index
        ];
    }

    shared[thread] = sum;

    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1)
    {
        if (thread < stride)
            shared[thread] += shared[thread + stride];

        __syncthreads();
    }

    if (thread == 0)
        gradients[filter] = shared[0] / 3600.0f;
}

__global__ void C1PooledErrorKernel(
    const float* c2Error,
    const float* c2Filters,
    float* c1PooledError)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= 7688)
        return;

    int channel = index / 3844;
    int position = index % 3844;

    int r = position / 62;
    int c = position % 62;

    float sum = 0.0f;

    for (int filter = 0; filter < 4; filter++)
    {
        for (int k = 0; k < 3; k++)
        {
            for (int l = 0; l < 3; l++)
            {
                int outputR = r - k;
                int outputC = c - l;

                if (outputR >= 0 &&
                    outputR < 60 &&
                    outputC >= 0 &&
                    outputC < 60)
                {
                    sum +=
                        c2Error[
                            filter * 3600 +
                                outputR * 60 +
                                outputC
                        ] *
                        c2Filters[
                            filter * 18 +
                                channel * 9 +
                                k * 3 +
                                l
                        ];
                }
            }
        }
    }

    c1PooledError[index] = sum;
}

__global__ void C1MaxPoolBackwardKernel(
    const float* input,
    const float* pooledError,
    float* error)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z;

    if (x >= 62 || y >= 62 || channel >= 2)
        return;

    int r = y * 2;
    int c = x * 2;

    int base = channel * 15376;

    int i0 = base + r * 124 + c;
    int i1 = base + r * 124 + c + 1;
    int i2 = base + (r + 1) * 124 + c;
    int i3 = base + (r + 1) * 124 + c + 1;

    float a = input[i0];
    float b = input[i1];
    float d = input[i2];
    float e = input[i3];

    float maxValue = a;
    int maxIndex = 0;

    if (b > maxValue)
    {
        maxValue = b;
        maxIndex = 1;
    }

    if (d > maxValue)
    {
        maxValue = d;
        maxIndex = 2;
    }

    if (e > maxValue)
    {
        maxValue = e;
        maxIndex = 3;
    }

    float errorValue =
        pooledError[
            channel * 3844 +
                y * 62 +
                x
        ];

    error[i0] = 0.0f;
    error[i1] = 0.0f;
    error[i2] = 0.0f;
    error[i3] = 0.0f;

    if (maxIndex == 0)
        error[i0] = errorValue;
    else if (maxIndex == 1)
        error[i1] = errorValue;
    else if (maxIndex == 2)
        error[i2] = errorValue;
    else
        error[i3] = errorValue;
}

__global__ void C1ReluBackwardKernel(
    const float* images,
    float* error)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= 30752)
        return;

    if (images[index] <= 0.0f)
        error[index] = 0.0f;
}

__global__ void C1GradientKernel(
    const float* error,
    const float* input,
    float* gradients)
{
    int weightIndex = blockIdx.x;

    if (weightIndex >= 50)
        return;

    __shared__ float shared[256];

    int thread = threadIdx.x;

    int filter = weightIndex / 25;
    int kernelIndex = weightIndex % 25;

    int k = kernelIndex / 5;
    int l = kernelIndex % 5;

    float sum = 0.0f;

    for (int index = thread; index < 15376; index += blockDim.x)
    {
        int r = index / 124;
        int c = index % 124;

        sum +=
            error[
                filter * 15376 +
                    index
            ] *
            input[
                (r + k) * 128 +
                    (c + l)
            ];
    }

    shared[thread] = sum;

    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1)
    {
        if (thread < stride)
            shared[thread] += shared[thread + stride];

        __syncthreads();
    }

    if (thread == 0)
        gradients[weightIndex] = shared[0] / 15376.0f;
}

__global__ void C1BiasGradientKernel(
    const float* error,
    float* gradients)
{
    int filter = blockIdx.x;

    if (filter >= 2)
        return;

    __shared__ float shared[256];

    int thread = threadIdx.x;

    float sum = 0.0f;

    for (int index = thread; index < 15376; index += blockDim.x)
    {
        sum += error[
            filter * 15376 +
                index
        ];
    }

    shared[thread] = sum;

    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1)
    {
        if (thread < stride)
            shared[thread] += shared[thread + stride];

        __syncthreads();
    }

    if (thread == 0)
        gradients[filter] = shared[0] / 15376.0f;
}

__global__ void UpdateWeightsKernel(
    float* weights,
    const float* gradients,
    int count,
    float learningRate)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count)
        return;

    weights[index] -=
        learningRate *
        gradients[index];
}

__global__ void UpdateBiasesKernel(
    float* c1Biases,
    float* c2Biases,
    const float* c1Gradients,
    const float* c2Gradients,
    float learningRate)
{
    int index = threadIdx.x;

    if (index < 2)
    {
        c1Biases[index] -=
            learningRate *
            c1Gradients[index];
    }

    if (index < 4)
    {
        c2Biases[index] -=
            learningRate *
            c2Gradients[index];
    }
}

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
    float C2F4Bias)
{
    CheckCUDA(cudaMalloc(&d_Input, 16384 * sizeof(float)));

    CheckCUDA(cudaMalloc(&d_C1Filters, 50 * sizeof(float)));
    CheckCUDA(cudaMalloc(&d_C1Images, 30752 * sizeof(float)));
    CheckCUDA(cudaMalloc(&d_C1ImagesPooled, 7688 * sizeof(float)));

    CheckCUDA(cudaMalloc(&d_C2Filters, 72 * sizeof(float)));
    CheckCUDA(cudaMalloc(&d_C2Images, 14400 * sizeof(float)));
    CheckCUDA(cudaMalloc(&d_C2ImagesPooled, 3600 * sizeof(float)));

    CheckCUDA(cudaMalloc(&d_FlatPixelValues, 3600 * sizeof(float)));

    CheckCUDA(cudaMalloc(&d_DenseWeights, 7200 * sizeof(float)));
    CheckCUDA(cudaMalloc(&d_OutputBiases, 2 * sizeof(float)));

    CheckCUDA(cudaMalloc(&d_Output, 2 * sizeof(float)));
    CheckCUDA(cudaMalloc(&d_Probabilities, 2 * sizeof(float)));
    CheckCUDA(cudaMalloc(&d_OutputError, 2 * sizeof(float)));
    CheckCUDA(cudaMalloc(&d_FlatError, 3600 * sizeof(float)));

    CheckCUDA(cudaMalloc(&d_C2Error, 14400 * sizeof(float)));
    CheckCUDA(cudaMalloc(&d_C2DLoss, 72 * sizeof(float)));
    CheckCUDA(cudaMalloc(&d_C2BiasLoss, 4 * sizeof(float)));

    CheckCUDA(cudaMalloc(&d_C1PooledError, 7688 * sizeof(float)));
    CheckCUDA(cudaMalloc(&d_C1Error, 30752 * sizeof(float)));
    CheckCUDA(cudaMalloc(&d_C1DLoss, 50 * sizeof(float)));
    CheckCUDA(cudaMalloc(&d_C1BiasLoss, 2 * sizeof(float)));

    CheckCUDA(cudaMalloc(&d_C1Biases, 2 * sizeof(float)));
    CheckCUDA(cudaMalloc(&d_C2Biases, 4 * sizeof(float)));

    float c1Biases[2] =
    {
        C1F1Bias,
        C1F2Bias
    };

    float c2Biases[4] =
    {
        C2F1Bias,
        C2F2Bias,
        C2F3Bias,
        C2F4Bias
    };

    CheckCUDA(cudaMemcpy(
        d_C1Filters,
        C1Filters,
        50 * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CheckCUDA(cudaMemcpy(
        d_C2Filters,
        C2Filters,
        72 * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CheckCUDA(cudaMemcpy(
        d_DenseWeights,
        DenseWeights,
        7200 * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CheckCUDA(cudaMemcpy(
        d_OutputBiases,
        OutputBiases,
        2 * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CheckCUDA(cudaMemcpy(
        d_C1Biases,
        c1Biases,
        2 * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CheckCUDA(cudaMemcpy(
        d_C2Biases,
        c2Biases,
        4 * sizeof(float),
        cudaMemcpyHostToDevice
    ));
}

void TrainImageCUDA(
    const float* input,
    int label,
    float DenseLearningRate,
    float ConvLearningRate,
    float* Probabilities,
    float& Loss,
    int& Prediction)
{
    CheckCUDA(cudaMemcpy(
        d_Input,
        input,
        16384 * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    dim3 threads2D(16, 16);

    dim3 blocksC1(
        (124 + 15) / 16,
        (124 + 15) / 16,
        2
    );

    C1ConvolutionKernel << <blocksC1, threads2D >> > (
        d_Input,
        d_C1Filters,
        d_C1Images,
        d_C1Biases
        );

    CheckCUDA(cudaGetLastError());

    dim3 blocksPool1(
        (62 + 15) / 16,
        (62 + 15) / 16,
        2
    );

    MaxPoolC1Kernel << <blocksPool1, threads2D >> > (
        d_C1Images,
        d_C1ImagesPooled
        );

    CheckCUDA(cudaGetLastError());

    dim3 blocksC2(
        (60 + 15) / 16,
        (60 + 15) / 16,
        4
    );

    C2ConvolutionKernel << <blocksC2, threads2D >> > (
        d_C1ImagesPooled,
        d_C2Filters,
        d_C2Images,
        d_C2Biases
        );

    CheckCUDA(cudaGetLastError());

    dim3 blocksPool2(
        (30 + 15) / 16,
        (30 + 15) / 16,
        4
    );

    MaxPoolC2Kernel << <blocksPool2, threads2D >> > (
        d_C2Images,
        d_C2ImagesPooled
        );

    CheckCUDA(cudaGetLastError());

    int blockSize = 256;

    FlattenKernel << <
        (3600 + blockSize - 1) / blockSize,
        blockSize
        >> > (
            d_C2ImagesPooled,
            d_FlatPixelValues
            );

    CheckCUDA(cudaGetLastError());

    DenseForwardKernel << <1, 32 >> > (
        d_FlatPixelValues,
        d_DenseWeights,
        d_OutputBiases,
        d_Output
        );

    CheckCUDA(cudaGetLastError());

    SoftmaxKernel << <1, 1 >> > (
        d_Output,
        d_Probabilities
        );

    CheckCUDA(cudaGetLastError());

    CheckCUDA(cudaMemcpy(
        Probabilities,
        d_Probabilities,
        2 * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    Prediction =
        Probabilities[0] > Probabilities[1]
        ? 0
        : 1;

    float probability = Probabilities[label];

    if (probability < 1e-10f)
        probability = 1e-10f;

    Loss = -logf(probability);

    OutputErrorKernel << <1, 2 >> > (
        d_Probabilities,
        d_OutputError,
        label
        );

    CheckCUDA(cudaGetLastError());

    FlatErrorKernel << <
        (3600 + blockSize - 1) / blockSize,
        blockSize
        >> > (
            d_OutputError,
            d_DenseWeights,
            d_FlatError
            );

    CheckCUDA(cudaGetLastError());

    DenseUpdateKernel << <
        (7200 + blockSize - 1) / blockSize,
        blockSize
        >> > (
            d_DenseWeights,
            d_OutputBiases,
            d_OutputError,
            d_FlatPixelValues,
            DenseLearningRate   // <-- was LearningRate
        );

    CheckCUDA(cudaGetLastError());

    dim3 blocksC2PoolBack(
        (30 + 15) / 16,
        (30 + 15) / 16,
        4
    );

    C2MaxPoolBackwardKernel << <
        blocksC2PoolBack,
        threads2D
        >> > (
            d_C2Images,
            d_FlatError,
            d_C2Error
            );

    CheckCUDA(cudaGetLastError());

    C2ReluBackwardKernel << <
        (14400 + blockSize - 1) / blockSize,
        blockSize
        >> > (
            d_C2Images,
            d_C2Error
            );

    CheckCUDA(cudaGetLastError());

    C2GradientKernel << <72, 256 >> > (
        d_C2Error,
        d_C1ImagesPooled,
        d_C2DLoss
        );

    CheckCUDA(cudaGetLastError());

    C2BiasGradientKernel << <4, 256 >> > (
        d_C2Error,
        d_C2BiasLoss
        );

    CheckCUDA(cudaGetLastError());

    C1PooledErrorKernel << <
        (7688 + blockSize - 1) / blockSize,
        blockSize
        >> > (
            d_C2Error,
            d_C2Filters,
            d_C1PooledError
            );

    CheckCUDA(cudaGetLastError());

    dim3 blocksC1PoolBack(
        (62 + 15) / 16,
        (62 + 15) / 16,
        2
    );

    C1MaxPoolBackwardKernel << <
        blocksC1PoolBack,
        threads2D
        >> > (
            d_C1Images,
            d_C1PooledError,
            d_C1Error
            );

    CheckCUDA(cudaGetLastError());

    C1ReluBackwardKernel << <
        (30752 + blockSize - 1) / blockSize,
        blockSize
        >> > (
            d_C1Images,
            d_C1Error
            );

    CheckCUDA(cudaGetLastError());

    C1GradientKernel << <50, 256 >> > (
        d_C1Error,
        d_Input,
        d_C1DLoss
        );

    CheckCUDA(cudaGetLastError());

    C1BiasGradientKernel << <2, 256 >> > (
        d_C1Error,
        d_C1BiasLoss
        );

    CheckCUDA(cudaGetLastError());

    UpdateWeightsKernel << <
        (72 + blockSize - 1) / blockSize,
        blockSize
        >> > (
            d_C2Filters,
            d_C2DLoss,
            72,
            ConvLearningRate   // <-- was LearningRate
            );

    CheckCUDA(cudaGetLastError());

    UpdateWeightsKernel << <
        (50 + blockSize - 1) / blockSize,
        blockSize
        >> > (
            d_C1Filters,
            d_C1DLoss,
            50,
            ConvLearningRate   // <-- was LearningRate
            );

    CheckCUDA(cudaGetLastError());

    UpdateBiasesKernel << <1, 4 >> > (
        d_C1Biases,
        d_C2Biases,
        d_C1BiasLoss,
        d_C2BiasLoss,
        ConvLearningRate   // <-- was LearningRate
        );

    CheckCUDA(cudaGetLastError());
}

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
    float& C2F4Bias)
{
    float c1Biases[2];
    float c2Biases[4];

    CheckCUDA(cudaDeviceSynchronize());

    CheckCUDA(cudaMemcpy(
        C1Filters,
        d_C1Filters,
        50 * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    CheckCUDA(cudaMemcpy(
        C2Filters,
        d_C2Filters,
        72 * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    CheckCUDA(cudaMemcpy(
        DenseWeights,
        d_DenseWeights,
        7200 * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    CheckCUDA(cudaMemcpy(
        OutputBiases,
        d_OutputBiases,
        2 * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    CheckCUDA(cudaMemcpy(
        c1Biases,
        d_C1Biases,
        2 * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    CheckCUDA(cudaMemcpy(
        c2Biases,
        d_C2Biases,
        4 * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    C1F1Bias = c1Biases[0];
    C1F2Bias = c1Biases[1];

    C2F1Bias = c2Biases[0];
    C2F2Bias = c2Biases[1];
    C2F3Bias = c2Biases[2];
    C2F4Bias = c2Biases[3];
}

void FreeCNN_CUDA()
{
    cudaFree(d_Input);

    cudaFree(d_C1Filters);
    cudaFree(d_C1Images);
    cudaFree(d_C1ImagesPooled);

    cudaFree(d_C2Filters);
    cudaFree(d_C2Images);
    cudaFree(d_C2ImagesPooled);

    cudaFree(d_FlatPixelValues);

    cudaFree(d_DenseWeights);
    cudaFree(d_OutputBiases);

    cudaFree(d_Output);
    cudaFree(d_Probabilities);
    cudaFree(d_OutputError);
    cudaFree(d_FlatError);

    cudaFree(d_C2Error);
    cudaFree(d_C2DLoss);
    cudaFree(d_C2BiasLoss);

    cudaFree(d_C1PooledError);
    cudaFree(d_C1Error);
    cudaFree(d_C1DLoss);
    cudaFree(d_C1BiasLoss);

    cudaFree(d_C1Biases);
    cudaFree(d_C2Biases);
}
