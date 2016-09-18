import numpy as np

'''
    A simple neural network with single hidden layer
'''

# Even no. of 1s = 1
training_data = np.array([
    [[0, 0, 1], 0],
    [[0, 1, 1], 1],
    [[1, 0, 1], 1],
    [[1, 1, 1], 0]
])


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_deriv(z):
    return z*(1-z)


def feedforward(weight, input_activation):
    z = np.dot(weight, input_activation)
    output_activation = sigmoid(z)
    return {"z":z, "output_activation":output_activation}


def output_error(expected_output, network_output):
    return expected_output - network_output

#
# def delta(activation, error):
#     return error * sigmoid_deriv(activation)
#
#
# def error(delta, weight):
#     return np.dot(np.transpose(weight), delta)

# Define weights from layer 0 to layer 1
num_input = 3
num_hidden = 5
num_output = 1

# The weights are from layer 0 to 1, but we reverse the indices i.e. use 10 instead of 01 because of convention and ease in matrix operations
weights10 = np.random.randn(num_hidden, num_input)
weights21 = np.random.randn(num_output, num_hidden)
learning_rate = 10

for training_loop in xrange(1, 100000):

    td_size = len(training_data)
    for td_index in xrange(0, td_size):

        '''
        PERFORM FEED FORWARD
        '''
        # From layer 0 (input layer) to layer 1
        # First column of training_data = input
        l0 = np.array([training_data[td_index, 0]]).T # 2X1
        l1_ff = feedforward(weights10, l0)
        l1 = l1_ff["output_activation"]  # num_hidden X 1
        # From layer 1 to layer 2 (output layer)
        l2_ff = feedforward(weights21, l1)
        l2 = l2_ff["output_activation"] # num_output X 1

        '''
        CALCULATE OUTPUT ERROR
        '''
        expected_output = training_data[td_index, 1] # First column of training_data = output
        l2_error = output_error(expected_output, l2) # num_output X 1
        if training_loop % 10000 == 0:
            print "Error = %f" %l2_error[0]

        '''
        PERFORM BACK PROPAGATION
        '''
        # First, let us propagate output layer's error to adjust weights between hidden layer and output layer
        l2_delta = l2_error * sigmoid_deriv(l2) # 1 X 1
        weights21 += learning_rate*(l1.dot(l2_delta.T)).T # num_output X num_hidden

        # Next propagate error from first hidden layer to adjust the weights between input layer and hidden layer
        l1_error = l2_delta.dot(weights21) # 1X5
        l1_delta = l1_error.T*sigmoid_deriv(l1) # 5X1
        weights10 += learning_rate*(l0.dot(l1_delta.T)).T # num_hidden X num_input


def output(input):
    l1 = feedforward(weights10, input)["output_activation"]
    l2 = feedforward(weights21, l1)["output_activation"]
    return l2

print output([0, 0, 1]) # Expect 0
print output([0, 1, 1]) # Expect 1
print output([1, 0, 1]) # Expect 1
print output([1, 1, 1]) # Expect 0