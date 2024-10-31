using ANN.ActivationFunctions;
using ANN.LossFunctions;
using System.Threading.Tasks;

namespace ANN.Layers
{
    /// <summary>
    /// Represents the output layer of a neural network, which computes the final output
    /// and loss for each neuron. This layer extends HiddenLayer by adding functionality
    /// specific to the network's output, including loss calculation and backpropagation.
    /// </summary>
    public class OutputLayer : HiddenLayer
    {
        /// <summary>
        /// Loss function used to measure the difference between real outputs and predicted outputs.
        /// </summary>
        public ILossFunction LossFunction { get; set; }

        /// <summary>
        /// Array to store the calculated loss for each neuron in the output layer.
        /// </summary>
        public float[] Loss { get; set; }

        /// <summary>
        /// Initializes a new instance of the OutputLayer class with specified perceptron size, input size, activation function, and loss function.
        /// </summary>
        /// <param name="perceptronSize">Number of neurons in this layer.</param>
        /// <param name="inputSize">Size of the input vector for each neuron.</param>
        /// <param name="activationFunction">Activation function for each neuron in the output layer.</param>
        /// <param name="lossFunction">Loss function to evaluate the error of predictions.</param>
        public OutputLayer(int perceptronSize, int inputSize, IActivationFunction activationFunction, ILossFunction lossFunction)
            : base(perceptronSize, inputSize, activationFunction)
        {
            LossFunction = lossFunction;
            Loss = new float[perceptronSize];
        }

        /// <summary>
        /// Calculates the loss and performs the backward pass for this output layer.
        /// </summary>
        /// <param name="inputs">The input values to this layer from the previous layer.</param>
        /// <param name="realOutputs">The actual target values to compare against the predicted outputs.</param>
        /// <returns>The current layer with updated gradients.</returns>
        public ILayer Backward(float[] inputs, float[] realOutputs)
        {
            // Calculate loss and error gradients for each neuron in the output layer
            Parallel.For(0, A.Length, i =>
            {
                // Compute the loss for the current neuron
                Loss[i] = LossFunction.Calculate(realOutputs[i], A[i]);

                // Calculate the gradient of the loss and activation function
                float dLoss = LossFunction.Derivative(realOutputs[i], A[i]);
                float dActivation = ActivationFunction.Derivative(A[i]);

                // Store the gradient of the error for backpropagation
                dE[i] = dLoss * dActivation;
            });
            // Calculate gradients for weights and biases using backpropagation
            Blas.cblas_sgemm(101, 111, 111, dE.Length, inputs.Length, 1, 1f, dE, 1, inputs, inputs.Length, 0f, dW, inputs.Length);
            Blas.cblas_saxpy(B.Length, 1f, dE, 1, dB, 1);

            return this;
        }
    }
}
