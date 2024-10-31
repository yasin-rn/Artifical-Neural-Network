# ANN.LossFunctions

`ANN.LossFunctions` is a C# namespace that provides implementations of common loss functions used in artificial neural networks (ANNs). Loss functions are crucial for evaluating the difference between the predicted output of a model and the actual target values, which guides the optimization process during training. The namespace includes the following loss functions:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Huber Loss

## Overview

Loss functions play an important role in neural network training by measuring how well the model's predictions match the actual values. By minimizing the loss, the model learns the underlying patterns of the data. Each loss function has its own characteristics and is suited for specific tasks.

### Implemented Loss Functions

1. **Mean Squared Error (MSE)**

   The MSE loss function calculates the average of the squared differences between predicted and actual values. It is commonly used for regression tasks.

   - **Formula**:
     
     \[ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_{i} - \hat{y}_{i})^2 \]

   - **Characteristics**:
     - Penalizes larger errors more heavily due to the squaring term.
     - Differentiable, making it suitable for gradient-based optimization.

2. **Mean Absolute Error (MAE)**

   The MAE loss function calculates the average of the absolute differences between predicted and actual values. It is robust to outliers.

   - **Formula**:
     
     \[ \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_{i} - \hat{y}_{i}| \]

   - **Characteristics**:
     - Treats all errors equally.
     - Less sensitive to outliers compared to MSE.

3. **Huber Loss**

   Huber Loss is a combination of MSE and MAE, aiming to be more robust to outliers while maintaining the differentiability of MSE. It behaves like MAE when the error is large and like MSE when the error is small.

   - **Formula**:
     
     \[ L_\delta(y, \hat{y}) = \begin{cases}
     \frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \le \delta, \\
     \delta \cdot (|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise}
     \end{cases} \]

   - **Characteristics**:
     - Provides a smooth transition between MSE and MAE.
     - Suitable for datasets containing outliers.

## ILossFunction Interface

All loss functions in `ANN.LossFunctions` implement the `ILossFunction` interface. This interface defines two essential methods:

- **Calculate(float real, float predict)**: Calculates the loss between the actual value (`real`) and the predicted value (`predict`).
- **Derivative(float real, float predict)**: Calculates the derivative of the loss function with respect to the predicted value. This derivative is crucial for backpropagation, allowing the model to adjust weights to minimize the loss.

### Example Usage

Below is an example of how to use the loss functions within this namespace:

```csharp
using ANN.LossFunctions;

public class NeuralNetworkExample
{
    public void Train()
    {
        ILossFunction lossFunction = new MeanSquaredError();
        float realValue = 5.0f;
        float predictedValue = 4.5f;

        float loss = lossFunction.Calculate(realValue, predictedValue);
        float gradient = lossFunction.Derivative(realValue, predictedValue);

        // Use the calculated loss and gradient for model optimization
    }
}
```

## Choosing the Right Loss Function

- **MSE** is suitable for scenarios where larger errors should be penalized heavily, such as regression tasks without significant outliers.
- **MAE** is ideal when outliers are present, as it does not overly penalize them compared to MSE.
- **Huber Loss** provides a balance between MSE and MAE, making it a good choice if your data has outliers but you still want to maintain differentiability.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Contributions

Contributions are welcome! Feel free to submit issues or pull requests for improvements or new features.
