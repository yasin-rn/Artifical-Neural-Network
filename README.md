# Neural Network Usage Example

This project contains a simple class that performs forward propagation, backpropagation, and weight updates using an artificial neural network (Neural Network). Below is a code snippet that demonstrates how to use the `NeuralNetwork` class.

## Contents

This README contains the following sections:
- Requirements
- Setup
- `Main` Function Explanation
- Neural Network Setup and Training

### Requirements

To use this project, you need the following libraries and tools:
- .NET Framework 4.7.2 or higher
- A C# compatible IDE (Visual Studio, VS Code, etc.)

### Setup

You can clone the project files or download them as a zip file to your computer. Then, in the project folder, you can use the following command to install dependencies and necessary libraries.

```bash
# Restore the .NET project
dotnet restore
```

Also, make sure the `experiences.csv` file is located in the root directory.

### Main Function Explanation

The `Main` function below uses data from an `experiences.csv` file to train and test an artificial neural network.

```csharp
static void Main(string[] args)
{
    var lines = File.ReadAllLines("experiences.csv");
    var cells = lines.Select(x => x.Split(',')).ToArray();

    var inputs = cells.Select(x => x.Take(6).Select(y => float.Parse(y)).ToArray()).ToArray();
    var outputs = cells.Select(x => x.Skip(6).Take(3).Select(y => float.Parse(y)).ToArray()).ToArray();

    NeuralNetwork neuralNetwork = new NeuralNetwork(inputSize: 6, outputSize: 3);

    neuralNetwork.AddHidden(perceptronSize: 32, new Linear());
    neuralNetwork.AddHidden(perceptronSize: 32, new Linear());
    neuralNetwork.AddHidden(perceptronSize: 32, new Linear());
    neuralNetwork.InitializeOutput(new Linear(), new MSE());

    neuralNetwork.Train(inputs, outputs, epoch: 6000, learningRate: 0.0001f, showStatusOnConsole: false);
}
```

#### Explanation of `Main` Function

1. **Loading Data from CSV File**:
   - The `File.ReadAllLines("experiences.csv")` method reads all lines from the `experiences.csv` file.
   - Each line is split by commas using `Split(',')` to create an array of cells.

2. **Preparing Input and Output Data**:
   - `inputs` are created by selecting the first 6 values from each row in the CSV file. These values represent the features used for training.
   - `outputs` are created by selecting the next 3 values from each row. These represent the target values that the network will try to predict.

3. **Creating the Neural Network**:
   - A new instance of the `NeuralNetwork` class is created with `inputSize` set to 6 and `outputSize` set to 3, matching the number of features and target values in the dataset.

4. **Adding Hidden Layers**:
   - Three hidden layers are added to the network, each containing 32 neurons and using a `Linear` activation function. The hidden layers are crucial for learning complex patterns in the data.

5. **Initializing the Output Layer**:
   - The output layer is initialized with a `Linear` activation function and an `MSE` (Mean Squared Error) loss function to measure the network's prediction error.

6. **Training the Neural Network**:
   - The `Train` method is called to train the network on the input and output data.
   - The training is performed for `6000` epochs with a learning rate of `0.0001f`. Setting `showStatusOnConsole` to `false` disables console output during training.

### Neural Network Setup and Training

1. **Loading Data**:
   The `experiences.csv` file is read using the `File.ReadAllLines()` method, and each line is split by a comma (`,`) into cells. These cells are used to create input and output data.

2. **Defining the Neural Network**:
   The neural network is defined using the `NeuralNetwork` class. The `inputSize` is set to 6, and the `outputSize` is set to 3 neurons.

3. **Adding Hidden Layers**:
   Three hidden layers are added to the neural network. Each layer has `32` neurons, and the `Linear` activation function is used.

4. **Defining the Output Layer**:
   The output layer is initialized with the `Linear` activation function and the `MSE` (Mean Squared Error) loss function.

5. **Training the Network**:
   The `Train` method is used to train the network. Training is performed for 6000 epochs, with a learning rate of `0.0001f`. Progress information is not shown on the console during training.

### Data File (experiences.csv)

The `experiences.csv` file should be in comma-separated (`.csv`) format. Each row should contain the input data (6 values) followed by the target output data (3 values). For example:

```
0.1,0.2,0.3,0.4,0.5,0.6,1.0,0.0,0.0
0.7,0.8,0.9,1.0,1.1,1.2,0.0,1.0,0.0
```

### Running the Project

You can run the application from the root directory with the following command:

```bash
dotnet run
```

This command will use the data from the `experiences.csv` file to train the neural network for the specified number of epochs.

### Notes

- You can replace the `Linear` activation function with other activation functions such as `ReLU`, `Sigmoid`, or `Tanh`.
- Ensure that the data in the `experiences.csv` file is correctly formatted and complete.
- The training duration depends on the size of the data and the complexity of the model. For large datasets, the training process may take a long time if the number of epochs is high.

### Contributing

If you would like to make any improvements or fix issues, please contribute to the project or create a `Pull Request`.

---
If you encounter any issues or would like to provide feedback, please feel free to reach out.

Happy coding!

