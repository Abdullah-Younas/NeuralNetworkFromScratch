#include <iostream>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <vector>
#include <cstdint> 

using namespace std;

vector<vector<int>> TrainingImagesData;
vector<int> TrainingImagesLabels;

vector<vector<int>> TestImagesData;
vector<int> TestImagesLabels;

vector<vector<double>> weightsInputHidden;
vector<vector<double>> weightsHiddenOutput;

vector<double> biasesHidden;
vector<double> biasesOutput;

uint32_t swapEndian32(uint32_t val) {
    return ((val >> 24) & 0xff) |
        ((val << 8) & 0xff0000) |
        ((val >> 8) & 0xff00) |
        ((val << 24) & 0xff000000);
}

void LoadTrainingDataMNIST()
{
    std::ifstream file("train-images-idx3-ubyte", std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "Failed to open the .idx file." << std::endl;
        return;
    }

    // 1. Read the 4-byte Magic Number
    uint8_t magic[4];
    file.read(reinterpret_cast<char*>(magic), 4);

    if (magic[0] != 0 || magic[1] != 0)
    {
        std::cerr << "Invalid .idx file format." << std::endl;
        return;
    }

    uint8_t dataType = magic[2];
    uint8_t numDimensions = magic[3];

    // 2. Read dimension sizes
    std::vector<uint32_t> dimSizes(numDimensions);
    size_t totalElements = 1;

    for (int i = 0; i < numDimensions; ++i)
    {
        uint32_t size;

        file.read(reinterpret_cast<char*>(&size), 4);

        size = swapEndian32(size);

        dimSizes[i] = size;
        totalElements *= size;

    }

    // 3. Read image data
    if (dataType == 0x08)
    {
        std::vector<uint8_t> dataBuffer(totalElements);

        file.read(
            reinterpret_cast<char*>(dataBuffer.data()),
            totalElements
        );

        if (file.gcount() != totalElements)
        {
            std::cerr << "Warning: Could not read all expected elements."
                << std::endl;
        }

        // Number of images
        size_t numberOfImages = dimSizes[0];

        // Pixels per image
        size_t pixelsPerImage = dimSizes[1] * dimSizes[2];

        // Create storage
        TrainingImagesData.resize(numberOfImages);

        for (size_t image = 0; image < numberOfImages; ++image)
        {
            TrainingImagesData[image].resize(pixelsPerImage);

            for (size_t pixel = 0; pixel < pixelsPerImage; ++pixel)
            {
                TrainingImagesData[image][pixel] =
                    static_cast<int>(
                        dataBuffer[image * pixelsPerImage + pixel]
                        );
            }
        }

    }
    else
    {
        std::cout << "Unsupported data type." << std::endl;
    }

    file.close();
}

void LoadTrainingDataLabelsMNIST()
{
    std::ifstream file(
        "train-labels-idx1-ubyte",
        std::ios::binary
    );

    if (!file.is_open())
    {
        std::cerr << "Error: Could not open label file."
            << std::endl;
        return;
    }

    // 1. Read header
    uint32_t magic_number = 0;
    uint32_t num_items = 0;

    file.read(
        reinterpret_cast<char*>(&magic_number),
        sizeof(magic_number)
    );

    file.read(
        reinterpret_cast<char*>(&num_items),
        sizeof(num_items)
    );

    // 2. Convert Big-Endian → Little-Endian
    magic_number = swapEndian32(magic_number);
    num_items = swapEndian32(num_items);

    // 3. Validate
    if (magic_number != 2049)
    {
        std::cerr << "Error: Invalid MNIST label file."
            << std::endl;
        return;
    }

    // 4. Resize our global label vector
    TrainingImagesLabels.resize(num_items);

    // 5. Read labels directly
    std::vector<uint8_t> labels(num_items);

    file.read(
        reinterpret_cast<char*>(labels.data()),
        num_items
    );

    // 6. Copy into TrainingImagesLabels
    for (size_t i = 0; i < num_items; ++i)
    {
        TrainingImagesLabels[i] =
            static_cast<int>(labels[i]);
    }


    file.close();
}

void LoadTestDataMNIST()
{
    ifstream file("t10k-images-idx3-ubyte", ios::binary);

    if (!file.is_open())
    {
        cout << "Failed to open test images!" << endl;
        return;
    }

    uint8_t magic[4];
    file.read(reinterpret_cast<char*>(magic), 4);

    uint8_t dataType = magic[2];
    uint8_t numDimensions = magic[3];

    vector<uint32_t> dimSizes(numDimensions);
    size_t totalElements = 1;

    for (int i = 0; i < numDimensions; i++)
    {
        uint32_t size;

        file.read(reinterpret_cast<char*>(&size), 4);

        size = swapEndian32(size);

        dimSizes[i] = size;
        totalElements *= size;
    }

    if (dataType == 0x08)
    {
        vector<uint8_t> dataBuffer(totalElements);

        file.read(
            reinterpret_cast<char*>(dataBuffer.data()),
            totalElements
        );

        size_t numberOfImages = dimSizes[0];
        size_t pixelsPerImage = dimSizes[1] * dimSizes[2];

        TestImagesData.resize(numberOfImages);

        for (size_t image = 0; image < numberOfImages; image++)
        {
            TestImagesData[image].resize(pixelsPerImage);

            for (size_t pixel = 0; pixel < pixelsPerImage; pixel++)
            {
                TestImagesData[image][pixel] =
                    static_cast<int>(
                        dataBuffer[image * pixelsPerImage + pixel]
                        );
            }
        }
    }

    file.close();

    cout << "Test images loaded: "
        << TestImagesData.size()
        << endl;
}

void LoadTestDataLabelsMNIST()
{
    ifstream file(
        "t10k-labels-idx1-ubyte",
        ios::binary
    );

    if (!file.is_open())
    {
        cout << "Failed to open test labels!" << endl;
        return;
    }

    uint32_t magic_number = 0;
    uint32_t num_items = 0;

    file.read(
        reinterpret_cast<char*>(&magic_number),
        sizeof(magic_number)
    );

    file.read(
        reinterpret_cast<char*>(&num_items),
        sizeof(num_items)
    );

    magic_number = swapEndian32(magic_number);
    num_items = swapEndian32(num_items);

    if (magic_number != 2049)
    {
        cout << "Invalid test label file!" << endl;
        return;
    }

    vector<uint8_t> labels(num_items);

    file.read(
        reinterpret_cast<char*>(labels.data()),
        num_items
    );

    TestImagesLabels.resize(num_items);

    for (size_t i = 0; i < num_items; i++)
    {
        TestImagesLabels[i] =
            static_cast<int>(labels[i]);
    }

    file.close();

    cout << "Test labels loaded: "
        << TestImagesLabels.size()
        << endl;
}

vector<double> NormalizeImage(const vector<int>& image)
{
    vector<double> normalized(784);

    for (int i = 0; i < 784; i++)
    {
        normalized[i] = image[i] / 255.0;
    }

    return normalized;
}

vector<double> NormalizeLabel(int labelTarget)
{
    vector<double> target(10, 0.0);

    target[labelTarget] = 1.0;

    return target;
}

double Sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

vector<double> Softmax(const vector<double>& logits,
    double temperature = 1.0)
{
    vector<double> probabilities(logits.size());

    double maxLogit =
        *max_element(logits.begin(), logits.end());

    double sum = 0.0;

    for (double x : logits)
    {
        sum += exp((x - maxLogit) / temperature);
    }

    for (int i = 0; i < logits.size(); i++)
    {
        probabilities[i] =
            exp((logits[i] - maxLogit) / temperature)
            / sum;
    }

    return probabilities;
}

void LoadWeights(string FileName, int input, int hidden, int output)
{
    ifstream file(FileName);

    if (!file)
    {
        cout << filesystem::current_path() << endl;
        cout << "Failed to open Weights.txt!" << endl;
        return;
    }

    vector<double> Nweights;
    double value;

    while (file >> value)
    {
        Nweights.push_back(value);
    }

    file.close();

    int requiredWeights =
        (input * hidden) +
        (hidden * output);

    if (Nweights.size() < requiredWeights)
    {
        cout << "Weights.txt does not contain enough weights!"
            << endl;

        cout << "Required: "
            << requiredWeights
            << endl;

        cout << "Found: "
            << Nweights.size()
            << endl;

        return;
    }

    // Input -> Hidden
    weightsInputHidden.resize(
        hidden,
        vector<double>(input)
    );

    int z = 0;

    for (int h = 0; h < hidden; h++)
    {
        for (int i = 0; i < input; i++)
        {
            weightsInputHidden[h][i] =
                Nweights[z++];

        }
    }

    // Hidden -> Output
    weightsHiddenOutput.resize(
        output,
        vector<double>(hidden)
    );

    for (int o = 0; o < output; o++)
    {
        for (int h = 0; h < hidden; h++)
        {
            weightsHiddenOutput[o][h] =
                Nweights[z++];
        }
    }
}

void LoadBiases(string FileName, int hidden, int output)
{
    ifstream file(FileName);

    if (!file)
    {
        cout << filesystem::current_path() << endl;
        cout << "Failed to open Biases.txt!" << endl;
        return;
    }

    vector<double> biases;
    double value;

    while (file >> value)
    {
        biases.push_back(value);
    }

    file.close();

    int requiredBiases =
        hidden + output;

    if (biases.size() < requiredBiases)
    {
        cout << "Biases.txt does not contain enough biases!"
            << endl;

        return;
    }

    // Hidden biases
    biasesHidden.resize(hidden);

    for (int h = 0; h < hidden; h++)
    {
        biasesHidden[h] =
            biases[h];
    }

    // Output biases
    biasesOutput.resize(output);

    for (int o = 0; o < output; o++)
    {
        biasesOutput[o] =
            biases[hidden + o];
    }
}

void TrainingMNISTNeuralNetwork(
    int NumOfEpochs,
    double LearningRate,
    vector<vector<double>>& weightsInputHidden,
    vector<vector<double>>& weightsHiddenOutput,
    int inputLayerSize,
    int hiddenLayerSize,
    int outputLayerSize,
    vector<double>& biasesHidden,
    vector<double>& biasesOutput)
{
    // Load MNIST only once
    LoadTrainingDataMNIST();
    LoadTrainingDataLabelsMNIST();


    for (int epoch = 0; epoch < NumOfEpochs; epoch++)
    {
        int correct = 0;
        double totalLoss = 0.0;
        int dataCount = 0;

        vector<double> hidden(hiddenLayerSize);
        vector<double> hiddenSigmoid(hiddenLayerSize);
        vector<double> hidden_Unit_Error(hiddenLayerSize);

        vector<double> outputs(outputLayerSize);
        vector<double> output_Unit_Error(outputLayerSize);

        for (int image = 0;
            image < TrainingImagesData.size();
            image++)
        {

            // Show progress every 1000 images
            if (image % 1000 == 0)
            {
                double progress =
                    (static_cast<double>(image) /
                        TrainingImagesData.size()) * 100.0;

                cout << "\rEpoch: "
                    << epoch
                    << " | Image: "
                    << image
                    << "/"
                    << TrainingImagesData.size()
                    << " | "
                    << progress
                    << "%"
                    << flush;
            }

            // Normalize image
            vector<double> inputs = NormalizeImage(TrainingImagesData[image]);

            // Get label
            int label = TrainingImagesLabels[image];

            // Convert label to one-hot target
            vector<double> target =
                NormalizeLabel(label);

            // Forward Pass
            // Input -> Hidden
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

                hiddenSigmoid[h] =
                    Sigmoid(hidden[h]);
            }


            // Hidden -> Output
            for (int o = 0; o < outputLayerSize; o++)
            {
                outputs[o] = biasesOutput[o];

                for (int h = 0; h < hiddenLayerSize; h++)
                {
                    outputs[o] +=
                        hiddenSigmoid[h] *
                        weightsHiddenOutput[o][h];
                }

            }
            outputs = Softmax(outputs, 1.0);

            int prediction =
                max_element(outputs.begin(), outputs.end())
                - outputs.begin();

            if (prediction == label)
            {
                correct++;
            }

            // Loss
            for (int o = 0; o < outputLayerSize; o++)
            {
                totalLoss +=
                    -target[o] * log(outputs[o] + 1e-15);
            }

            dataCount++;


            // Backpropagation
            // Output errors
            for (int o = 0; o < outputLayerSize; o++)
            {
                output_Unit_Error[o] =
                    target[o] - outputs[o];
            }


            // Hidden errors
            for (int h = 0; h < hiddenLayerSize; h++)
            {
                hidden_Unit_Error[h] = 0.0;

                for (int o = 0; o < outputLayerSize; o++)
                {
                    hidden_Unit_Error[h] +=
                        weightsHiddenOutput[o][h] *
                        output_Unit_Error[o];
                }

                hidden_Unit_Error[h] *=
                    hiddenSigmoid[h] *
                    (1.0 - hiddenSigmoid[h]);
            }


            // Hidden -> Output weights
            for (int o = 0; o < outputLayerSize; o++)
            {
                for (int h = 0; h < hiddenLayerSize; h++)
                {
                    double delta =
                        LearningRate *
                        output_Unit_Error[o] *
                        hiddenSigmoid[h];

                    weightsHiddenOutput[o][h] += delta;
                }

                // Output bias
                double delta_output_bias =
                    LearningRate *
                    output_Unit_Error[o];

                biasesOutput[o] +=
                    delta_output_bias;
            }


            // Input -> Hidden weights
            for (int h = 0; h < hiddenLayerSize; h++)
            {
                for (int i = 0; i < inputLayerSize; i++)
                {
                    double delta =
                        LearningRate *
                        hidden_Unit_Error[h] *
                        inputs[i];

                    weightsInputHidden[h][i] += delta;
                }

                // Hidden bias
                double delta_hidden_bias =
                    LearningRate *
                    hidden_Unit_Error[h];

                biasesHidden[h] +=
                    delta_hidden_bias;
            }
        }

        cout << endl;

        // Epoch information
        if (epoch % 1 == 0)
        {
            double averageLoss =
                totalLoss / dataCount;

            double accuracy =
                100.0 * correct / dataCount;

            cout << "Epoch: "
                << epoch
                << " | Average Cross Entropy: "
                << averageLoss
                << " | Accuracy: "
                << accuracy
                << endl;
        }
    }
}

void PredictingMode(
    int inputLayerSize,
    int hiddenLayerSize,
    int outputLayerSize,
    vector<double>& biasesHidden,
    vector<double>& biasesOutput)
{
    int imageNumber;

    cout << "Enter image number (0 - "
        << TrainingImagesData.size() - 1
        << "): ";

    cin >> imageNumber;

    if (imageNumber < 0 ||
        imageNumber >= TrainingImagesData.size())
    {
        cout << "Invalid image number!" << endl;
        return;
    }

    // Get image
    vector<double> inputs =
        NormalizeImage(TrainingImagesData[imageNumber]);

    vector<double> hidden(hiddenLayerSize);
    vector<double> hiddenSigmoid(hiddenLayerSize);

    vector<double> outputs(outputLayerSize);


    // =========================
    // Input -> Hidden
    // =========================

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

        hiddenSigmoid[h] =
            Sigmoid(hidden[h]);
    }


    // =========================
    // Hidden -> Output
    // =========================

    for (int o = 0; o < outputLayerSize; o++)
    {
        outputs[o] =
            biasesOutput[o];

        for (int h = 0; h < hiddenLayerSize; h++)
        {
            outputs[o] +=
                hiddenSigmoid[h] *
                weightsHiddenOutput[o][h];
        }
    }

    outputs = Softmax(outputs, 1.0);


    // =========================
    // Find highest probability
    // =========================

    int predictedDigit = 0;

    for (int o = 1; o < outputLayerSize; o++)
    {
        if (outputs[o] > outputs[predictedDigit])
        {
            predictedDigit = o;
        }
    }


    // =========================
    // Results
    // =========================

    cout << endl;

    cout << "Predicted Digit: "
        << predictedDigit
        << endl;

    cout << "Actual Digit: "
        << TrainingImagesLabels[imageNumber]
        << endl;

    cout << endl;

    cout << "Probabilities:" << endl;

    for (int o = 0; o < outputLayerSize; o++)
    {
        cout << o
            << ": "
            << outputs[o]
            << endl;
    }
}

void TestMNISTNeuralNetwork(
    vector<vector<double>>& weightsInputHidden,
    vector<vector<double>>& weightsHiddenOutput,
    int inputLayerSize,
    int hiddenLayerSize,
    int outputLayerSize,
    vector<double>& biasesHidden,
    vector<double>& biasesOutput)
{
    LoadTestDataMNIST();
    LoadTestDataLabelsMNIST();

    int correct = 0;

    for (int image = 0;
        image < TestImagesData.size();
        image++)
    {
        vector<double> inputs =
            NormalizeImage(TestImagesData[image]);

        // Forward Pass
        vector<double> hidden(hiddenLayerSize);
        vector<double> hiddenSigmoid(hiddenLayerSize);

        for (int h = 0; h < hiddenLayerSize; h++)
        {
            hidden[h] = 0.0;

            for (int i = 0; i < inputLayerSize; i++)
            {
                hidden[h] += inputs[i] * weightsInputHidden[h][i];
            }

            hidden[h] += biasesHidden[h];

            hiddenSigmoid[h] = Sigmoid(hidden[h]);
        }

        vector<double> outputs(outputLayerSize);

        for (int o = 0; o < outputLayerSize; o++)
        {
            outputs[o] =
                biasesOutput[o];

            for (int h = 0; h < hiddenLayerSize; h++)
            {
                outputs[o] +=
                    hiddenSigmoid[h] *
                    weightsHiddenOutput[o][h];
            }
        }

        outputs = Softmax(outputs, 1.0);

        // Find highest probability

        int predictedDigit = 0;

        for (int o = 1; o < outputLayerSize; o++)
        {
            if (outputs[o] > outputs[predictedDigit])
            {
                predictedDigit = o;
            }
        }

        // Compare with actual
        int actualDigit = TestImagesLabels[image];

        if (predictedDigit == actualDigit)
        {
            correct++;
        }

        // Progress
        if (image % 100 == 0)
        {
            double progress =
                (static_cast<double>(image) /
                    TestImagesData.size()) * 100.0;

            cout << "\rTesting: " << image << "/" << TestImagesData.size() << " | " << progress << "%" << flush;
        }
    }

    double accuracy = (static_cast<double>(correct) / TestImagesData.size()) * 100.0;

    cout << endl;

    cout << "MNIST TEST RESULTS" << endl;

    cout << "Correct: "
        << correct
        << "/"
        << TestImagesData.size()
        << endl;

    cout << "Incorrect: "
        << TestImagesData.size() - correct
        << endl;

    cout << "Accuracy: "
        << accuracy
        << "%"
        << endl;
}

int main()
{

    // Layers
    int input = 784; // K
    int hidden = 128; // N
    int output = 10;

    int Mode;

    // Load weights
    LoadWeights("Weights.txt", input, hidden, output);

    // Biases
    LoadBiases("Biases.txt", hidden, output);

    // Learning rate
    double learning_Rate = 0.05;

    // Training
    const int epochs = 10;
    
    cout << "Choose a Mode(Press 0 for Training the Network OR Press 1 for Predicting): ";
    cin >> Mode;

    if (Mode == 0) {

        TrainingMNISTNeuralNetwork(
            epochs,
            learning_Rate,
            weightsInputHidden,
            weightsHiddenOutput,
            input,
            hidden,
            output,
            biasesHidden,
            biasesOutput
        );

        // Save updated weights
        ofstream weightFile("Weights.txt");

        if (!weightFile)
        {
            cout << "Failed to save Weights.txt!" << endl;
            return 1;
        }

        // Input -> Hidden
        for (int h = 0; h < hidden; h++)
        {
            for (int i = 0; i < input; i++)
            {
                weightFile << weightsInputHidden[h][i] << endl;
            }
        }

        // Hidden -> Output
        for (int o = 0; o < output; o++)
        {
            for (int h = 0; h < hidden; h++)
            {
                weightFile << weightsHiddenOutput[o][h] << endl;
            }
        }

        weightFile.close();

        ofstream biasFile("Biases.txt");

        if (!biasFile)
        {
            cout << "Failed to save Biases.txt!" << endl;
            return 1;
        }

        // Hidden biases
        for (int h = 0; h < hidden; h++)
        {
            biasFile << biasesHidden[h] << endl;
        }

        // Output biases
        for (int o = 0; o < output; o++)
        {
            biasFile << biasesOutput[o] << endl;
        }

        biasFile.close();

    }
    else if (Mode == 1)
    {
        TestMNISTNeuralNetwork(
            weightsInputHidden,
            weightsHiddenOutput,
            input,
            hidden,
            output,
            biasesHidden,
            biasesOutput
        );
    }
    else {
        cout << "Unknown Mode";
    }
    return 0;
}

