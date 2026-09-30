#!/bin/bash

g++ -std=c++17 "Neural Network.cpp" -o "Neural Network"

if [ $? -eq 0 ]; then
    echo "Build successful."
    ./"Neural Network"
else
    echo "Build failed."
fi
