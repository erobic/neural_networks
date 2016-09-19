# Neural Networks

This is where I implement networks and tweak things to improve their performances

## network.py
A neural network with n-layers and selectable cost and activation functions

### Sample program:
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
