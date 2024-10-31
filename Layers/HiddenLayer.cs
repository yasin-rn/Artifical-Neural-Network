using ANN.ActivationFunctions;
using System;
using System.Linq;
using System.Threading.Tasks;

namespace ANN.Layers
{
    /// <summary>
    /// Represents a hidden layer in a neural network, capable of performing forward and backward propagation.
    /// The layer uses an activation function and applies weights and biases to its inputs.
    /// </summary>
    public class HiddenLayer : ILayer
    {
        public int PerceptronSize { get; set; } // Number of neurons in this layer
        public int InputSize { get; set; } // Number of inputs each neuron receives
        public IActivationFunction ActivationFunction { get; set; } // Activation function for this layer

        public float[] W { get; set; } // Weights for each input connection
        public float[] A { get; set; } // Activations (output values) for each neuron after applying the activation function
        public float[] B { get; set; } // Bias values for each neuron

        public float[] dE { get; set; } // Error gradients with respect to activations for each neuron
        public float[] dW { get; set; } // Gradients of the loss with respect to the weights
        public float[] dB { get; set; } // Gradients of the loss with respect to the biases

        static Random Random = new Random(); // Static random instance for initializing weights


        /// <summary>
        /// Initializes a new instance of the HiddenLayer class with the specified number of perceptrons, inputs, and activation function.
        /// </summary>
        /// <param name="perceptronSize">Number of neurons in this layer.</param>
        /// <param name="inputSize">Size of the input vector for each neuron.</param>
        /// <param name="activationFunction">Activation function to be used by each neuron.</param>
        public HiddenLayer(int perceptronSize, int inputSize, IActivationFunction activationFunction)
        {
            PerceptronSize = perceptronSize;
            InputSize = inputSize;
            ActivationFunction = activationFunction;

            // Initialize arrays for activations, biases, and gradients
            A = new float[PerceptronSize];
            B = new float[PerceptronSize];
            W = Enumerable.Range(0, inputSize * perceptronSize).Select(x => (float)(((Random.NextDouble() * 2) - 1))).ToArray();
           
            dW = new float[inputSize * perceptronSize];
            dB = new float[perceptronSize];
            dE = new float[perceptronSize];
        }

        /// <summary>
        /// Performs the forward pass, calculating the output activations for this layer.
        /// </summary>
        /// <param name="inputs">The input values to the layer.</param>
        /// <returns>The activations after applying weights, biases, and the activation function.</returns>
        public float[] Forward(float[] inputs)
        {
            // Matrix-vector multiplication to calculate weighted sums
            Blas.cblas_sgemv(101, 111, A.Length, inputs.Length, 1f, W, inputs.Length, inputs, 1, 0f, A, 1);

            // Apply activation function and biases
            Parallel.For(0, A.Length, i =>
            {
                A[i] = ActivationFunction.Activate(A[i] + B[i]);

            });
            return A;
        }

        /// <summary>
        /// Performs the backward pass, calculating gradients for weights, biases, and activations.
        /// </summary>
        /// <param name="nextLayer">The next layer in the network, used to calculate gradients.</param>
        /// <param name="inputs">The input values to the layer.</param>
        /// <returns>The current layer with updated gradients.</returns>
        public ILayer Backward(ILayer nextLayer, float[] inputs)
        {
            // Calculate error gradients for the current layer based on the next layer's errors
            Blas.cblas_sgemv(102, 111, dE.Length, nextLayer.dE.Length, 1f, nextLayer.W, dE.Length, nextLayer.dE, 1, 0f, dE, 1);

            // Adjust gradients by the derivative of the activation function
            Parallel.For(0, dE.Length, i =>
            {
                dE[i] *= ActivationFunction.Derivative(A[i]);
            });

            // Calculate weight and bias gradients for this layer
            Blas.cblas_sgemm(101, 111, 111, dE.Length, inputs.Length, 1, 1f, dE, 1, inputs, inputs.Length, 0f, dW, inputs.Length);
            Blas.cblas_saxpy(B.Length, 1f, dE, 1, dB, 1);

            return this;
        }

        /// <summary>
        /// Updates the layer's weights and biases using the calculated gradients and learning rate.
        /// </summary>
        /// <param name="lr">Learning rate used to scale the update.</param>
        public void Update(float lr)
        {
            // Update weights and biases by scaling gradients with the learning rate
            Blas.cblas_saxpy(W.Length, -lr, dW, 1, W, 1);
            Blas.cblas_saxpy(B.Length, -lr, dB, 1, B, 1);

            // Reset gradients after each update
            ResetGradients();
        }

        /// <summary>
        /// Resets the gradients for weights, biases, and errors to zero after an update.
        /// </summary>
        void ResetGradients()
        {
            dW = new float[dW.Length];
            dB = new float[dB.Length];
            dE = new float[dE.Length];
        }
    }
}
