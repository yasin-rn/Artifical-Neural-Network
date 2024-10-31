using ANN.ActivationFunctions;
using System;

namespace ANN.ActivationFunctions
{
    /// <summary>
    /// Implements the Tanh (hyperbolic tangent) activation function for neural network layers.
    /// </summary>
    public class Tanh : IActivationFunction
    {
        /// <summary>
        /// Applies the Tanh activation function to the given input value.
        /// The Tanh function outputs values between -1 and 1.
        /// </summary>
        /// <param name="value">The input value to activate.</param>
        /// <returns>The result of the Tanh function.</returns>
        public float Activate(float value)
        {
            return MathF.Tanh(value);  // Tanh formula: (e^x - e^(-x)) / (e^x + e^(-x))
        }

        /// <summary>
        /// Computes the derivative of the Tanh function based on the activation output.
        /// The derivative is used for backpropagation in training.
        /// </summary>
        /// <param name="activationValue">The activated value, output from the Tanh function.</param>
        /// <returns>The derivative of the Tanh function, calculated as 1 - activationValue^2.</returns>
        public float Derivative(float activationValue)
        {
            return 1 - (activationValue * activationValue);  // Tanh derivative: 1 - f(x)^2
        }
    }
}
