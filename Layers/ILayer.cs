using ANN.ActivationFunctions;

namespace ANN.Layers
{
    /// <summary>
    /// Interface for a neural network layer, defining the structure and operations
    /// needed for forward propagation and parameter updates.
    /// </summary>
    public interface ILayer
    {
        /// <summary>
        /// The number of perceptrons (neurons) in the layer.
        /// </summary>
        int PerceptronSize { get; set; }

        /// <summary>
        /// The size of the input vector to the layer.
        /// </summary>
        int InputSize { get; set; }

        /// <summary>
        /// The activation function used by neurons in this layer.
        /// </summary>
        IActivationFunction ActivationFunction { get; set; }

        /// <summary>
        /// Weight matrix for connections between neurons, represented as a flat array.
        /// </summary>
        float[] W { get; set; }

        /// <summary>
        /// The output (activations) of the layer after applying the activation function.
        /// </summary>
        float[] A { get; set; }

        /// <summary>
        /// Bias values for each neuron in the layer.
        /// </summary>
        float[] B { get; set; }

        /// <summary>
        /// Error gradients with respect to the activations for each neuron in the layer.
        /// Used in backpropagation to update weights and biases.
        /// </summary>
        float[] dE { get; set; }

        /// <summary>
        /// Gradients of the loss with respect to the weights in the layer.
        /// </summary>
        float[] dW { get; set; }

        /// <summary>
        /// Gradients of the loss with respect to the biases in the layer.
        /// </summary>
        float[] dB { get; set; }

        /// <summary>
        /// Computes the forward pass for this layer, taking input values and
        /// returning the output after applying weights, biases, and the activation function.
        /// </summary>
        /// <param name="inputs">The input values to the layer.</param>
        /// <returns>The output values after the forward pass.</returns>
        float[] Forward(float[] inputs);

        /// <summary>
        /// Updates the layer’s weights and biases based on the learning rate and gradients.
        /// </summary>
        /// <param name="lr">The learning rate used to scale the gradient updates.</param>
        void Update(float lr);
    }
}
