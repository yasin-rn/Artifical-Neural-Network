# ANN.Layers Module

The `ANN.Layers` module is part of an artificial neural network (ANN) library, providing core functionalities for creating and managing neural network layers. The module defines two main types of layers: `HiddenLayer` and `OutputLayer`, along with an `ILayer` interface that standardizes the layer structure. The purpose of these classes is to support building a feedforward neural network capable of learning from data through forward and backward propagation.

## Table of Contents
- [ILayer Interface](#ilayer-interface)
- [HiddenLayer Class](#hiddenlayer-class)
- [OutputLayer Class](#outputlayer-class)
- [Usage](#usage)
- [Dependencies](#dependencies)

## ILayer Interface

The `ILayer` interface defines the contract for a layer in the neural network. This includes properties and methods that any implementing class should have:

- **Properties**:
  - `PerceptronSize`: The number of perceptrons (neurons) in the layer.
  - `InputSize`: The size of the input vector to the layer.
  - `ActivationFunction`: The activation function used by neurons in the layer.
  - `W`, `B`, `A`: Weight matrix, bias values, and output activations respectively.
  - `dE`, `dW`, `dB`: Error gradients used for backpropagation.
- **Methods**:
  - `Forward(float[] inputs)`: Performs forward propagation, returning the output after applying weights, biases, and the activation function.
  - `Update(float lr)`: Updates the weights and biases using the gradients and learning rate.

## HiddenLayer Class

The `HiddenLayer` class implements the `ILayer` interface and represents a fully connected layer in the neural network. The class has functionalities to:

- **Forward Pass**: Computes the activations by multiplying input vectors with weights, adding biases, and applying the activation function.
- **Backward Pass**: Calculates the gradients for weights and biases using the chain rule to propagate errors to earlier layers.
- **Update Parameters**: Adjusts the weights and biases based on the calculated gradients and a specified learning rate.

### Key Methods

- **`Forward(float[] inputs)`**: Takes an input vector and computes the layer's output using matrix-vector multiplication and the activation function.
- **`Backward(ILayer nextLayer, float[] inputs)`**: Uses the next layer's gradients to compute this layer's error gradients.
- **`Update(float lr)`**: Updates weights and biases by scaling gradients with the learning rate.

## OutputLayer Class

The `OutputLayer` class extends the `HiddenLayer` class and includes additional functionality to compute the final output and calculate the loss. It represents the output layer of a neural network, where the final predictions are generated, and the loss is computed for backpropagation.

### Key Features

- **Loss Calculation**: Uses a provided loss function to calculate the difference between the predicted output and the actual target values.
- **Gradient Calculation**: Calculates the loss gradients for backpropagation.

### Key Methods

- **`Backward(float[] inputs, float[] realOutputs)`**: Computes the gradients of the loss for each output neuron and updates the error gradients for backpropagation.

## Usage

### Initializing Layers

To create a hidden layer, use the `HiddenLayer` constructor:

```csharp
var hiddenLayer = new HiddenLayer(perceptronSize: 10, inputSize: 5, activationFunction: new ReLU());
```

To create an output layer, use the `OutputLayer` constructor:

```csharp
var outputLayer = new OutputLayer(perceptronSize: 10, inputSize: 5, activationFunction: new Softmax(), lossFunction: new CrossEntropyLoss());
```

### Forward and Backward Propagation

To perform forward propagation through a layer:

```csharp
float[] inputs = { 1.0f, 0.5f, -0.3f, 0.7f, 0.2f };
float[] activations = hiddenLayer.Forward(inputs);
```

To perform backward propagation and update the layer parameters:

```csharp
hiddenLayer.Backward(nextLayer: outputLayer, inputs: inputs);
hiddenLayer.Update(lr: 0.01f);
```

## Dependencies

The `ANN.Layers` module relies on:

- **`ANN.ActivationFunctions`**: Activation functions like ReLU, Sigmoid, etc., are defined in this namespace.
- **`ANN.LossFunctions`**: Loss functions like CrossEntropy, MeanSquaredError, etc., are defined in this namespace.
- **`System.Linq` and `System.Threading.Tasks`**: Used for operations such as initializing arrays and parallel processing.
- **`Blas`**: Used for efficient linear algebra operations like matrix-vector multiplication (`cblas_sgemv`) and vector scaling (`cblas_saxpy`).

Ensure that all dependencies are included in your project for successful compilation and execution.

