using System;

namespace ANN.LossFunctions
{
    /// <summary>
    /// Implementation of the Huber Loss function.
    /// Huber Loss is a combination of Mean Squared Error (MSE) and Mean Absolute Error (MAE),
    /// providing a balance between the sensitivity of MSE to outliers and the robustness of MAE.
    /// </summary>
    public class HuberLoss : ILossFunction
    {
        /// <summary>
        /// The delta value determines the threshold where the loss function transitions 
        /// from a quadratic to linear form.
        /// </summary>
        private readonly float _delta;

        /// <summary>
        /// Initializes a new instance of the HuberLoss class with a specified delta.
        /// </summary>
        /// <param name="delta">The threshold at which the Huber Loss function switches 
        /// between MSE and MAE behavior. Default is 1.0.</param>
        public HuberLoss(float delta = 1.0f)
        {
            _delta = delta;
        }

        /// <summary>
        /// Calculates the Huber Loss between the actual and predicted values.
        /// </summary>
        /// <param name="real">The actual target value.</param>
        /// <param name="predict">The predicted value from the model.</param>
        /// <returns>The computed Huber Loss value. Uses quadratic behavior for small errors, 
        /// and linear behavior for larger errors, controlled by delta.</returns>
        public float Calculate(float real, float predict)
        {
            float error = real - predict;
            if (Math.Abs(error) <= _delta)
                return 0.5f * (float)Math.Pow(error, 2);
            else
                return _delta * (Math.Abs(error) - 0.5f * _delta);
        }

        /// <summary>
        /// Calculates the derivative of the Huber Loss function with respect to the predicted value.
        /// </summary>
        /// <param name="real">The actual target value.</param>
        /// <param name="predict">The predicted value from the model.</param>
        /// <returns>The derivative of the Huber Loss, which is error for small errors (quadratic behavior),
        /// or delta * sign(error) for large errors (linear behavior).</returns>
        public float Derivative(float real, float predict)
        {
            float error = predict - real;
            if (Math.Abs(error) <= _delta)
                return error;
            else
                return _delta * (error > 0 ? 1 : -1);
        }
    }
}
