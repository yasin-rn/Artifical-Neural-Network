namespace ANN.ActivationFunctions
{
    /// <summary>
    /// Interface defining activation functions for neural network layers.
    /// </summary>
    public interface IActivationFunction
    {
        /// <summary>
        /// Activates a given input value according to the specific activation function.
        /// </summary>
        /// <param name="value">The input value to activate.</param>
        /// <returns>The activated output as a float.</returns>
        float Activate(float value);

        /// <summary>
        /// Calculates the derivative of the activation function 
        /// based on the activation value (output of the activation function).
        /// This is often used during the backpropagation process.
        /// </summary>
        /// <param name="activationValue">The value obtained after activation.</param>
        /// <returns>The derivative of the activation function as a float.</returns>
        float Derivative(float activationValue);
    }
}
