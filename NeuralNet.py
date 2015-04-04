__author__ = 'Aaron'

# Adapted from: http://stackoverflow.com/questions/15395835/simple-multi-layer-neural-network-implementation
# and: http://www.ai-junkie.com/ann/evolved/

import random
import math

BIAS = -1


class Neuron:
    """
    Represents a single node in the neural net, described by its inputs and applied weights.
    """

    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.weights = [random.uniform(0, 1) for _ in range(num_inputs + 1)]  # additional weight for bias

    def sum(self, inputs):
        """
        Given inputs to the neuron, return the weighted sum, not including bias.
        :param inputs: inputs to the neuron
        :return: weighted sum
        """
        return sum(val * self.weights[i] for i, val in enumerate(inputs))

    def set_weights(self, weights):
        self.weights = weights


class NeuronLayer:
    """
    Represents a layer of neurons, described by the number of neurons and inputs to each
    """

    def __init__(self, num_neurons, num_inputs_per_neuron):
        self.num_neurons = num_neurons
        self.neurons = [Neuron(num_inputs_per_neuron) for _ in range(num_neurons)]


class NeuralNet:
    """
    Represents the entire neural net, including the layer descriptions and functions to operate on them.
    """

    def __init__(self, num_inputs, num_outputs, num_hidden_layers, num_neurons_per_hl):
        self.num_inputs = num_inputs  # Number of inputs to the neural net
        self.num_outputs = num_outputs  # Number of outputs from the network
        self.num_hidden_layers = num_hidden_layers  # Number of hidden layers in the network
        self.num_neurons_per_hl = num_neurons_per_hl  # Number of nodes in each hidden layer

        self._num_weights = None

        if num_hidden_layers > 0:
            # First layer
            self.layers = [NeuronLayer(num_neurons_per_hl, num_inputs)]  # TODO understand this initialization

            # Hidden layer(s)
            self.layers += [NeuronLayer(num_neurons_per_hl, num_neurons_per_hl) for _ in range(0, num_hidden_layers)]

            # Output layer with inputs from hidden
            self.layers += [NeuronLayer(num_outputs, num_neurons_per_hl)]
        else:
            # Single layer
            self.layers = [NeuronLayer(num_outputs, num_inputs)]

    def get_weights(self):
        """
        Return a list of all the weights in the neural network
        :return: list containing all the neurons' weights
        """
        weights = []

        for layer in self.layers:
            for neuron in layer.neurons:
                weights += neuron.weights

        return weights

    @property
    def num_weights(self):
        if self._num_weights is None:
            self._num_weights = 0
            for layer in self.layers:
                for neuron in layer.neurons:
                    self._num_weights += neuron.num_inputs + 1  # add bias weight
        return self._num_weights

    def set_weights(self, weights):
        """
        Put the input weights list into the network in order.
        :param weights: list of weights to put into the network
        """
        assert len(weights) == self.num_weights, "Incorrect number of input weights."

        stop = 0

        # Translates linear weights list to the neural net
        for layer in self.layers:
            for neuron in layer.neurons:
                start, stop = stop, stop + (neuron.num_inputs + 1)
                neuron.set_weights(weights[start:stop])
        return self

    def update(self, inputs):
        """
        Given a set of inputs, calculate the outputs from the neural network
        :param inputs: the inputs to feed into the network
        :return: list of output values
        """
        assert len(inputs) == self.num_inputs, "Incorrect number of input weights."

        # Store outputs for each layer
        outputs = []

        # Process each layer
        for layer in self.layers:
            for neuron in layer.neurons:
                # Add products of weights and inputs to weighted bias
                total = neuron.sum(inputs) + neuron.weights[-1] * BIAS

                # Filter through sigmoid function
                outputs.append(NeuralNet.sigmoid(total))
            inputs = outputs

        return outputs

    @staticmethod
    def sigmoid(activation, response=1):
        """
        Implements the sigmoid function.
        :param activation: value
        :param response: optional parameter to tune response
        :return: filtered value
        """
        try:
            return 1 / (1 + math.e ** (-activation / response))
        except OverflowError:
            return float("inf")
