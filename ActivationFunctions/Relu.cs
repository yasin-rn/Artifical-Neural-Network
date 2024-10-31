using ANN.ActivationFunctions;
using System;

/// <summary>
/// Implementation of the ReLU (Rectified Linear Unit) activation function.
/// This function outputs the input directly if it is positive; otherwise, it outputs zero.
/// ReLU is widely used in neural networks due to its ability to introduce non-linearity and reduce the likelihood of vanishing gradients.
/// </summary>
public class ReLU : IActivationFunction
{
    /// <summary>
    /// Applies the ReLU activation function to the input value.
    /// </summary>
    /// <param name="value">The input value to be activated.</param>
    /// <returns>The activated value. Returns the input if it is positive; otherwise, returns zero.</returns>
    public float Activate(float value)
    {
        return Math.Max(0, value);
    }

    /// <summary>
    /// Calculates the derivative of the ReLU activation function based on the activated value.
    /// </summary>
    /// <param name="activationValue">The activated value for which to calculate the derivative.</param>
    /// <returns>The derivative of the function. Returns 1 if the activation value is positive; otherwise, returns 0.</returns>
    public float Derivative(float activationValue)
    {
        return activationValue > 0 ? 1 : 0;
    }
}
