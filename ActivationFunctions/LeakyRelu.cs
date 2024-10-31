using ANN.ActivationFunctions;

/// <summary>
/// Implementation of the LeakyReLU (Leaky Rectified Linear Unit) activation function.
/// This function introduces a small slope for negative values to prevent them from being completely zeroed out.
/// </summary>
public class LeakyReLU : IActivationFunction
{
    /// <summary>
    /// Alpha parameter that determines the slope for negative values.
    /// This is typically set to a small positive number (e.g., 0.01).
    /// </summary>
    private readonly float _alpha;

    /// <summary>
    /// Initializes a new instance of the LeakyReLU class with the specified alpha parameter.
    /// </summary>
    /// <param name="alpha">The slope applied to negative input values. Default is 0.01.</param>
    public LeakyReLU(float alpha = 0.01f)
    {
        _alpha = alpha;
    }

    /// <summary>
    /// Applies the LeakyReLU activation function to the input value.
    /// </summary>
    /// <param name="value">The input value to activate.</param>
    /// <returns>The activated output value. If the input is positive, it returns the input;
    /// if negative, it returns alpha * input.</returns>
    public float Activate(float value)
    {
        return value > 0 ? value : _alpha * value;
    }

    /// <summary>
    /// Calculates the derivative of the LeakyReLU activation function given an activated value.
    /// </summary>
    /// <param name="activationValue">The activated value to calculate the derivative for.</param>
    /// <returns>The derivative of the function. Returns 1 if the input is positive; 
    /// otherwise, returns alpha.</returns>
    public float Derivative(float activationValue)
    {
        return activationValue > 0 ? 1 : _alpha;
    }
}
