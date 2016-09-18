import numpy as np

'''
    A simple neural network with single hidden layer
'''

num_input = 3
num_hidden = 10
num_output = 4
# Class 0 = Odd no. of 1s
# Class 1 = Even no. of 1s
# Classes 2 and 3 should never occur
training_data = np.array([
    [[0, 0, 1], [1, 0, 0, 0]],
    [[0, 1, 1], [0, 1, 0, 0]],
    [[1, 0, 1], [0, 1, 0, 0]],
    [[1, 1, 1], [1, 0, 0, 0]]
])


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return z*(1-z)

def cost(expected_output, actual_output):
    return np.sum(np.nan_to_num(expected_output*np.log(actual_output)
                  + (1-expected_output)*(np.log(1-actual_output))))


def feedforward(weight, input_activation):
    z = np.dot(weight, input_activation)
    output_activation = sigmoid(z)
    return {"z":z, "output_activation":output_activation}


def output_error(expected_output, network_output):
    return expected_output - network_output

# Define weights
weights10 = np.random.randn(num_hidden, num_input)
weights21 = np.random.randn(num_output, num_hidden)
learning_rate = 3

for training_loop in xrange(1, 100000):

    td_size = len(training_data)
    current_batch_error = 0
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
        expected_output = np.reshape(training_data[td_index, 1], (num_output, 1)) # First column of training_data = output. no. of outputs X
        l2_error = expected_output - l2 # no. of outputs X 1
        current_batch_error+= np.abs(l2_error[0])

        '''
        PERFORM BACK PROPAGATION
        '''
        # First, let us propagate output layer's error to adjust weights between hidden layer and output layer
        weights21 += learning_rate*(l1.dot(l2_error.T * sigmoid_prime(l2).T)).T # hidden * 1

        # Next propagate error from first hidden layer to adjust the weights between input layer and hidden layer
        l1_error = weights21.T.dot(l2_error)  # hidden X 1 unconf
        weights10 += learning_rate*(l0.dot(l1_error.T * sigmoid_prime(l1).T)).T # num_hidden X num_input

    if training_loop%5000==0:
        print "Error : %f" %current_batch_error
        print "Cost: "
        print cost(np.reshape(training_data[-1, 1], (num_output, 1)), l2)

def output(input):
    l1 = feedforward(weights10, input)["output_activation"]
    l2 = feedforward(weights21, l1)["output_activation"]
    return l2

print output([0, 0, 1]) # Expect 0
print output([0, 1, 1]) # Expect 1
print output([1, 0, 1]) # Expect 1
print output([1, 1, 1]) # Expect 0