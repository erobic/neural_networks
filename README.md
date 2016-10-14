# Neural Networks

Implementation of Neural network with Backpropagation and Stochastic Gradient Descent. Cost and activation functions can be switched easily.

### Usage:
```python
import network
model = network.Network([784, 30, 10])
model.execute(training_data,
            learning_rate=3,
            epochs=100,
            batch_size=20,
            regularization_param=0.01,
            validation_data=validation_data)
```
