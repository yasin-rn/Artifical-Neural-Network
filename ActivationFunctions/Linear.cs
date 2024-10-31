using ANN.ActivationFunctions;

/// <summary>
/// Implementation of the Linear activation function.
/// This function is an identity function where the output is equal to the input, 
/// commonly used when no transformation on the input is desired.
/// </summary>
public class Linear : IActivationFunction
{
    /// <summary>
    /// Applies the Linear activation function to the input value.
    /// </summary>
    /// <param name="value">The input value to activate.</param>
    /// <returns>The output, which is the same as the input value.</returns>
    public float Activate(float value)
    {
        return value;
    }

    /// <summary>
    /// Calculates the derivative of the Linear activation function.
    /// </summary>
    /// <param name="activationValue">The activated value for which to calculate the derivative.</param>
    /// <returns>The derivative of the function, which is always 1 for a linear function.</returns>
    public float Derivative(float activationValue)
    {
        return 1;
    }
}
