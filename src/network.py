import numpy as np


class CrossEntropyCost(object):
    @staticmethod
    def delta(expected_output, network_output):
        return network_output - expected_output

    @staticmethod
    def cost(expected_output, network_output, n):
        return (1/n)*(expected_output*np.log(network_output) + (1-expected_output)*np.log(1-network_output))


class SigmoidActivation(object):
    @staticmethod
    def activate(z):
        return 1.0/(1.0+np.exp(-z))

    @staticmethod
    def deriv(z):
        return z*(1-z)


class Network(object):

    def __init__(self, layer_sizes, cost=CrossEntropyCost, activation=SigmoidActivation):
        '''
        Constructs a neural network
        :param layer_sizes: Each element represents the number of neurons in that layer. Example [10, 20, 30, 5] means network with 10 input neurons, 20 neurons in first hidden layer, 30 neurons in second hidden layer and 5 neurons in output layer
        :param cost: The type of cost function. It is CrossEntropyCost by default.
        :param activation: The type of activation function. It is SigmoidActivation by default.
        '''
        self.layer_sizes = layer_sizes
        self.cost = cost
        self.activation = activation

        self.num_layers = len(layer_sizes)
        # Initialize weights
        # Each row is a (#neurons in l+1, #neurons in l) matrix with random weights
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        # Each row is a (#neurons in l+1, 1) matrix
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]

    def feedforward(self, input_data):
        """
        Computes z = wx+b and a = activation(z) starting from input layer upto output layer
        :param input_data:
        :return:
        """
        zs = []# input_data = num_l1 X 1
        activations = [input_data]
        for l in xrange(0, self.num_layers-1):
            z = np.dot(self.weights[l], activations[l]) + self.biases[l]
            zs.append(z)
            activation = self.activation.activate(z)
            activations.append(activation)
        return activations

    def batches(self, training_data, batch_size, num_data):
        """
        Randomizes the training_data and gives an array whose each element has a maximum of batch_size number of data elements.
        :param training_data:
        :param batch_size:
        :param num_data:
        :return: An array whose each element has a maximum of batch_size number of data elements
        """
        np.random.shuffle(training_data)
        return [training_data[x: x + batch_size] for x in xrange(0, num_data, batch_size)]

    def backprop(self, expected_output, activations):
        """
        Performs the backpropagation
        Calculates error at the output layer, propagtes this error backwards to compute adjustments for weights and biases of the network
        :param expected_output:
        :param activations:
        :return: A dictionary with 'weight_adjustments' and 'bias_adjustments'
        """
        output_error = self.cost.delta(expected_output, activations[-1])# num_output X 1
        errors = [output_error]
        weight_deltas = []
        weight_adjustments = []

        for curr_layer in xrange(self.num_layers-1, 0, -1):
            # Output error should be multiplied by derivative of activation function to get delta value
            weight_delta = errors[0]*self.activation.deriv(activations[curr_layer]) # num_neurons in (curr_layer) X 1
            weight_deltas.insert(0, weight_delta)

            prev_layer_error = self.weights[curr_layer - 1].T.dot(weight_delta) # num_neurons in (curr_layer-1) X 1
            errors.insert(0, prev_layer_error)

            weight_adjustment = weight_delta.dot(activations[curr_layer-1].T)# num_neurons in curr_layer X num_neurons in curr_layer-1
            weight_adjustments.insert(0, weight_adjustment)

        # bias adjustments = delta
        bias_adjustments = weight_deltas
        return {"weight_adjustments": weight_adjustments, "bias_adjustments": bias_adjustments}

    def adjust_weights_and_biases(self, weight_adjustments, bias_adjustments, learning_rate, regularization_param, num_data, batch_size):
        """
        Updates the weights and biases of the network with provided matrices for adjustments. L2 regularization is used to subdue overfitting.
        """
        for l in xrange(0, self.num_layers-1):
            self.weights[l] = (1-(learning_rate*regularization_param/num_data))*self.weights[l] \
                               - (learning_rate/batch_size)*weight_adjustments[l]
            self.biases[l] = self.biases[l] - (learning_rate/batch_size)*bias_adjustments[l]

    def execute(self, training_data, learning_rate, epochs=100, batch_size=10, regularization_param=1.0, validation_data=None):
        """
        Trains the neural network with given training_data for given number of epochs
        """
        num_data = len(training_data)
        for epoch_num in xrange(0, epochs):
            print "Executing epoch: %d..." %epoch_num
            batches = self.batches(training_data, batch_size, num_data)
            for batch in batches:
                batch_weight_adjustments = None
                batch_bias_adjustments = None
                num_elems_in_batch = len(batch)
                for data in batch:
                    # Feed forward entire network. The activations will be then used for backpropagation
                    input = data[0] #First column is the input data
                    activations = self.feedforward(input)
                    expected_output = data[1]
                    adjustments = self.backprop(expected_output, activations)
                    if batch_weight_adjustments is None:
                        batch_weight_adjustments = adjustments["weight_adjustments"]
                        batch_bias_adjustments = adjustments["bias_adjustments"]
                    else:
                        batch_weight_adjustments = np.add(batch_weight_adjustments, adjustments["weight_adjustments"])
                        batch_bias_adjustments = np.add(batch_bias_adjustments, adjustments["bias_adjustments"])

                self.adjust_weights_and_biases(batch_weight_adjustments, batch_bias_adjustments, learning_rate, regularization_param, num_data, num_elems_in_batch)
            if validation_data is not None:
                validation_data_accuracy = self.accuracy(validation_data)
                print "Validation accuracy of: %d/%d" % (validation_data_accuracy, len(validation_data))

    def accuracy(self, test_data):
        """
        Calculates the number of data items for which the prediction was accurate
        :return: Number of data items for which the prediction was accurate
        """
        total_num = len(test_data)
        correct_num = 0
        for i in xrange(0, total_num):
            curr_data = test_data[i]
            input = curr_data[0]
            expected_output = np.argmax(curr_data[1])
            activations = self.feedforward(input)
            network_output = np.argmax(activations[-1])
            if expected_output == network_output:
                correct_num += 1
        return correct_num






