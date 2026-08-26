#pragma once
#include <vector>

// Plain host function - callable from main.cpp (MSVC) or main.cu (nvcc)
void RunMatMul(std::vector<double>* inputs, std::vector<std::vector<double>>& weights, std::vector<double>* hidden, int M, int N, int K);