# ANN.ActivationFunctions README

## Overview

`ANN.ActivationFunctions` is a library that provides a set of commonly used activation functions for neural network layers. Activation functions play a critical role in the training of neural networks by introducing non-linearities, which enable the network to learn complex relationships in data. This library currently supports the following activation functions:

- LeakyReLU
- Linear
- ReLU
- Sigmoid
- Tanh

Each activation function is implemented as a class that adheres to the `IActivationFunction` interface, which defines methods for activation and derivative calculation.

## Activation Functions Included

### 1. LeakyReLU

The LeakyReLU activation function is a variant of the ReLU function that allows a small, non-zero gradient when the unit is not active. This addresses the issue of "dead neurons" often found in ReLU.

- **Equation**:
  - For ,&#x20;
  - For , , where  is a small constant (e.g., 0.01).

### 2. Linear

The Linear activation function is simply an identity function that returns the input value unchanged. It is often used in the output layer for regression tasks.

- **Equation**:&#x20;

### 3. ReLU (Rectified Linear Unit)

ReLU is the most popular activation function used in modern neural networks, especially deep networks. It helps to solve the vanishing gradient problem.

- **Equation**:
  - For ,&#x20;
  - For ,&#x20;

### 4. Sigmoid

The Sigmoid function maps the input into a range between 0 and 1. It is particularly useful in binary classification tasks.

- **Equation**:&#x20;

### 5. Tanh (Hyperbolic Tangent)

The Tanh function is similar to the Sigmoid function, but its output ranges from -1 to 1, which makes it a better choice in practice as it centers the data.

- **Equation**:&#x20;

## IActivationFunction Interface

The `IActivationFunction` interface defines the blueprint for each activation function. It includes two methods:

1. **Activate(float value)**

   - Takes an input value and applies the activation function, returning the output.

2. **Derivative(float activationValue)**

   - Takes the output of the activation function and returns the derivative. This is crucial for backpropagation during the training of neural networks.

## Usage

You can use the activation functions by implementing the `IActivationFunction` interface in your neural network layers. For example:

```csharp
using ANN.ActivationFunctions;

class Example
{
    static void Main(string[] args)
    {
        IActivationFunction activation = new Sigmoid();
        float inputValue = 1.0f;
        float activatedValue = activation.Activate(inputValue);

        // Output the activated value
        Console.WriteLine($"Activated Value: {activatedValue}");

        // Compute and output the derivative
        float derivativeValue = activation.Derivative(activatedValue);
        Console.WriteLine($"Derivative Value: {derivativeValue}");
    }
}
```

### Example: Neural Network Usage

Below is an example of how to create and train a simple neural network using the `ANN.ActivationFunctions` library:

```csharp
using ANN.ActivationFunctions;

static void Main(string[] args)
{
    NeuralNetwork neuralNetwork = new NeuralNetwork(4, 2);

    neuralNetwork.AddHidden(16, new Tanh());
    neuralNetwork.AddHidden(16, new Tanh());
    neuralNetwork.InitializeOutput(new Sigmoid(), new MSE());
}
```

In this example, a neural network is created with 4 input neurons, 2 output neurons, and two hidden layers with 16 neurons each. The `Tanh` activation function is used for the hidden layers, while the `Sigmoid` function is used for the output layer. The network is then trained with a given input and real output using backpropagation and an update step.

## Installation

Currently, this library is designed for internal use. You can include the source files in your project or convert it into a reusable package.
