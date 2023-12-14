import numpy as np
import matplotlib.pyplot as plt
from typing import List

class Neuron:
    pass

def eval_activation_relu(x):
    return np.maximum(x, 0)

def eval_least_squares(y, yhat):
    return np.sum(np.square(y - yhat))

def gaussian(x, mean, variance):
    return (1 / (np.sqrt(variance * 2 * np.pi))) * np.exp(-0.5 * np.power((x-mean), 2)/variance)

def model(x, input_neurons, hidden_neurons, output_neurons):
    # not vectorised

    outputs = []
    for xval in x:
        hidden_activations = []
        for i, h in enumerate(hidden_neurons):
            pre_activation = [h.bias + c.weight * xval for c in h.connections]
            post_activation = eval_activation_relu(pre_activation)
            hidden_activations.append(post_activation)

        value = output_neurons[0].bias + np.sum([c.weight * hidden_activations[j] for j, c in enumerate(output_neurons[0].connections)])
        outputs.append(value)

    return outputs


class Connection:
    neuron: Neuron = None
    weight: float = 0

class Neuron:
    bias: float = 0
    connections: List[Connection] = []

    def __init__(self):
        self.connections = []


if __name__ == "__main__":
    x = np.linspace(-5, 5, 100)
    y = -2 * x + 5


    # plt.plot(x, y)
    # plt.show()

    input_size = 1
    hidden_size = 100
    output_size = 1
    batch_size = 100

    # initialize neuron sets
    input_neurons = []
    for i in range(input_size):
        input_neurons.append(Neuron())
    hidden_neurons = []
    for i in range(hidden_size):
        hidden_neurons.append(Neuron())
    output_neurons = []
    for i in range(output_size):
        output_neurons.append(Neuron())

    variance_weights = 2 / hidden_size
    random_weight_samples = np.random.normal(0, np.sqrt(variance_weights), hidden_size)

    # generate map of output -> middle and middle -> input
    # +
    # initialize weights based on He initialization
    # biases can be set to zero, but weights must have a specific variance
    # of 2 / Dh
    # we can generate weights by creating a gaussian distribution with
    # a mean of zero and the specified variance, and then sampling within the
    # space randomly
    for o in output_neurons:
        for h in hidden_neurons:
            connection = Connection()
            connection.neuron = h
            connection.weight = np.random.normal(0, np.sqrt(4 / (100 + 1)))
            o.connections.append(connection)

    for h in hidden_neurons:
        for i in input_neurons:
            connection = Connection()
            connection.neuron = i
            connection.weight = np.random.normal(0, np.sqrt(4 / (1 + 100)))
            h.connections.append(connection)


    # forward pass to find the loss
    x_batch = x[:batch_size]
    predictions = model(x_batch, input_neurons, hidden_neurons, output_neurons)
    loss = eval_least_squares(predictions, y)
    a = 0
    # backward pass to find the gradients for SGD
    

    # SGD
