namespace ANN.LossFunctions
{
    /// <summary>
    /// Interface defining methods for calculating loss and its derivative in neural networks.
    /// Loss functions measure the difference between predicted and actual values, 
    /// and are essential for guiding model optimization during training.
    /// </summary>
    public interface ILossFunction
    {
        /// <summary>
        /// Calculates the loss given the actual (real) and predicted values.
        /// </summary>
        /// <param name="real">The actual target value.</param>
        /// <param name="predict">The predicted value from the model.</param>
        /// <returns>The computed loss, representing the difference between real and predicted values.</returns>
        float Calculate(float real, float predict);

        /// <summary>
        /// Calculates the derivative of the loss function with respect to the predicted value.
        /// This is essential for backpropagation, as it indicates how to adjust the model's predictions 
        /// to minimize the loss.
        /// </summary>
        /// <param name="real">The actual target value.</param>
        /// <param name="predict">The predicted value from the model.</param>
        /// <returns>The derivative of the loss function with respect to the prediction.</returns>
        float Derivative(float real, float predict);
    }
}
