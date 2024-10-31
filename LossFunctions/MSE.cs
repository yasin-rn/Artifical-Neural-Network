using System;

namespace ANN.LossFunctions
{
    /// <summary>
    /// Implementation of the Mean Squared Error (MSE) loss function.
    /// MSE measures the average squared difference between actual (real) and predicted values.
    /// This function is commonly used for regression problems as it heavily penalizes larger errors.
    /// </summary>
    public class MSE : ILossFunction
    {
        /// <summary>
        /// Calculates the Mean Squared Error (MSE) loss between the real and predicted values.
        /// </summary>
        /// <param name="real">The actual target value.</param>
        /// <param name="predict">The predicted value from the model.</param>
        /// <returns>The computed MSE, calculated as the square of the difference between real and predicted values.</returns>
        public float Calculate(float real, float predict)
        {
            return (float)Math.Pow(real - predict, 2);
        }

        /// <summary>
        /// Calculates the derivative of the MSE loss function with respect to the predicted value.
        /// This derivative is used to adjust the model parameters during training.
        /// </summary>
        /// <param name="real">The actual target value.</param>
        /// <param name="predict">The predicted value from the model.</param>
        /// <returns>The derivative of the MSE, which is 2 * (predict - real), indicating the direction 
        /// and magnitude of change needed to minimize the error.</returns>
        public float Derivative(float real, float predict)
        {
            return 2 * (predict - real);
        }
    }
}
