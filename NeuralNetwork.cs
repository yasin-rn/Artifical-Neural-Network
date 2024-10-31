using ANN.ActivationFunctions;
using ANN.LossFunctions;
using ANN.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Xml.Linq;
using System.Threading.Tasks;

namespace ANN
{
    /// <summary>
    /// Represents a fully connected neural network with a configurable number of hidden layers and an output layer.
    /// The network supports forward propagation, backpropagation, and parameter updates.
    /// </summary>
    public class NeuralNetwork
    {
        /// <summary>
        /// List of hidden layers in the network.
        /// </summary>
        List<HiddenLayer> HiddenLayers { get; set; }

        /// <summary>
        /// Output layer of the network, responsible for final predictions.
        /// </summary>
        OutputLayer OutputLayer { get; set; }

        /// <summary>
        /// Number of input features.
        /// </summary>
        int InputSize { get; set; }

        /// <summary>
        /// Number of output neurons in the output layer.
        /// </summary>
        int OutputSize { get; set; }

        /// <summary>
        /// Flag to indicate if the output layer has been initialized.
        /// </summary>
        bool OutputInitialized { get; set; }

        /// <summary>
        /// Stores the network's input data for use during backpropagation.
        /// </summary>
        float[] Input { get; set; }

        /// <summary>
        /// Initializes a new neural network instance with the specified input and output sizes.
        /// </summary>
        /// <param name="inputSize">Number of input features.</param>
        /// <param name="outputSize">Number of output neurons in the output layer.</param>
        public NeuralNetwork(int inputSize, int outputSize)
        {
            HiddenLayers = new List<HiddenLayer>();
            InputSize = inputSize;
            OutputSize = outputSize;
        }

        /// <summary>
        /// Performs a forward pass through the network, computing the output for the given input.
        /// </summary>
        /// <param name="input">Input data array.</param>
        /// <returns>Output of the network after the forward pass.</returns>
        public float[] Forward(float[] input)
        {
            Input = input;
            var Output = input;

            // Pass input through each hidden layer
            for (int i = 0; i < HiddenLayers.Count; i++)
            {
                Output = HiddenLayers[i].Forward(Output);
            }

            // Pass through the output layer
            Output = OutputLayer.Forward(Output);

            return Output;
        }

        /// <summary>
        /// Performs backpropagation to compute gradients based on the actual output and the target values.
        /// </summary>
        /// <param name="real">The target output values.</param>
        /// <returns>Loss for the current output layer.</returns>
        public float[] Backward(float[] real)
        {
            // Calculate gradients for the output layer
            var nextLayer = OutputLayer.Backward(HiddenLayers.Last().A, real);

            // Backpropagate through hidden layers in reverse order
            for (int i = HiddenLayers.Count - 1; i > 0; i--)
            {
                nextLayer = HiddenLayers[i].Backward(nextLayer, HiddenLayers[i - 1].A);
            }
            HiddenLayers[0].Backward(nextLayer, Input);

            return OutputLayer.Loss;
        }

        /// <summary>
        /// Updates weights and biases in each layer based on computed gradients and learning rate.
        /// </summary>
        /// <param name="lr">Learning rate for the update step. Default is 0.001.</param>
        public void Update(float lr = 0.001f)
        {
            // Update hidden layers
            Parallel.For(0, HiddenLayers.Count, (i) =>
            {
                HiddenLayers[i].Update(lr);
            });

            // Update output layer
            OutputLayer.Update(lr);
        }

        /// <summary>
        /// Trains the neural network on provided inputs and outputs for a specified number of epochs.
        /// </summary>
        /// <param name="inputs">A 2D array of input data. Each row is a training sample, and each column is a feature.</param>
        /// <param name="outputs">A 2D array of target output data corresponding to the inputs. Each row represents the expected output for the training sample.</param>
        /// <param name="epoch">The number of times the network will process all training data (full training cycles).</param>
        /// <param name="showStatusOnConsole">If true, a loading bar and loss information are displayed on the console during training.</param>
        public void Train(float[][] inputs, float[][] outputs, int epoch, float learningRate, bool showStatusOnConsole)
        {

            for (int iEpoch = 0; iEpoch < epoch; iEpoch++)
            {
                float totalLoss = 0;
                
                for (int iData = 0; iData < inputs.Length; iData++)
                {
                    var a = Forward(inputs[iData]);
                    totalLoss += Backward(outputs[iData]).Sum();
                    Update(learningRate);

                }
                if (showStatusOnConsole)
                {
                    int progress = (int)((iEpoch + 1) / (float)epoch * 50);
                    Console.Write($"Epoch {iEpoch + 1}/{epoch}: [{new string('=', progress)} {new string(' ', 50 - progress)}] {progress * 2}% - Iteration {iEpoch}/{epoch} Loss: {(totalLoss / inputs.Length).ToString("0.000")}");
                    Console.SetCursorPosition(0, 0);
                }
            }
        }


        /// <summary>
        /// Adds a hidden layer to the network.
        /// </summary>
        /// <param name="perceptronSize">Number of neurons in the hidden layer.</param>
        /// <param name="activationFunction">Activation function for this layer.</param>
        /// <exception cref="Exception">Thrown if the output layer has already been initialized.</exception>
        public void AddHidden(int perceptronSize, IActivationFunction activationFunction)
        {
            if (!OutputInitialized)
            {
                int inputSize = HiddenLayers.Count == 0 ? InputSize : HiddenLayers.Last().PerceptronSize;
                HiddenLayers.Add(new HiddenLayer(perceptronSize, inputSize, activationFunction));
            }
            else
            {
                throw new Exception("Add hidden before initializing output.");
            }
        }

        /// <summary>
        /// Initializes the output layer with a specified activation function and loss function.
        /// </summary>
        /// <param name="outputActivation">Activation function for the output layer.</param>
        /// <param name="outputLossFunction">Loss function for calculating errors in the output layer.</param>
        public void InitializeOutput(IActivationFunction outputActivation, ILossFunction outputLossFunction)
        {
            OutputLayer = new OutputLayer(OutputSize, HiddenLayers.Last().PerceptronSize, outputActivation, outputLossFunction);
            OutputInitialized = true;
        }
    }
}
