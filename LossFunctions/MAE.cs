using System;

namespace ANN.LossFunctions
{
    /// <summary>
    /// Implementation of the Mean Absolute Error (MAE) loss function.
    /// MAE measures the average absolute difference between the actual (real) and predicted values.
    /// It is commonly used in regression tasks and is less sensitive to large outliers than MSE.
    /// </summary>
    public class MAE : ILossFunction
    {
        /// <summary>
        /// Calculates the Mean Absolute Error (MAE) loss between the real and predicted values.
        /// </summary>
        /// <param name="real">The actual target value.</param>
        /// <param name="predict">The predicted value from the model.</param>
        /// <returns>The computed MAE, which is the absolute difference between real and predicted values.</returns>
        public float Calculate(float real, float predict)
        {
            return Math.Abs(real - predict);
        }

        /// <summary>
        /// Calculates the derivative of the MAE loss function with respect to the predicted value.
        /// The derivative is either 1 or -1, depending on whether the prediction is greater or less than the actual value.
        /// </summary>
        /// <param name="real">The actual target value.</param>
        /// <param name="predict">The predicted value from the model.</param>
        /// <returns>The derivative of the MAE: returns 1 if the prediction is greater than the real value; otherwise, returns -1.</returns>
        public float Derivative(float real, float predict)
        {
            return predict > real ? 1 : -1;
        }
    }
}
