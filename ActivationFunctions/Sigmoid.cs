using ANN.ActivationFunctions;
using System;

namespace ANN.ActivationFunctions
{
    /// <summary>
    /// Implements the Sigmoid activation function, commonly used in neural networks.
    /// </summary>
    public class Sigmoid : IActivationFunction
    {
        /// <summary>
        /// Applies the Sigmoid activation function to the given input value.
        /// </summary>
        /// <param name="value">The input value to activate.</param>
        /// <returns>The result of the Sigmoid function, a value between 0 and 1.</returns>
        public float Activate(float value)
        {
            return 1 / (1 + MathF.Exp(-value));  // Sigmoid formula: 1 / (1 + e^(-x))
        }

        /// <summary>
        /// Computes the derivative of the Sigmoid function based on the activation output.
        /// This is useful during backpropagation.
        /// </summary>
        /// <param name="activationValue">The activated value, output from the Sigmoid function.</param>
        /// <returns>The derivative of the Sigmoid function, used for gradient calculation.</returns>
        public float Derivative(float activationValue)
        {
            return activationValue * (1 - activationValue);  // Sigmoid derivative: f(x) * (1 - f(x))
        }
    }
}
