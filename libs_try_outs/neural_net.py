# %%

import numpy as np


class NeuralNet():
    def __init__(self):
        self.synaptic_weights = 2 * np.random.random((3, 1))-1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1-x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.test(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error *
                                 self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def test(self, inputs):

        input = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output


print(2 * np.random.random((3, 1))-1)
neural_net = NeuralNet()

print("Random synaptic weights:")
print(neural_net.synaptic_weights)

training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

training_outputs = np.array([[0, 1, 1, 0]]).T

neural_net.train(training_inputs, training_outputs, 10000)

print("Synaptic weights after training:")
print(neural_net.synaptic_weights)

A = 0
B = 1
C = 0

print("New input data: ", A, B, C)
print("Output data:")
print(neural_net.test(np.array([A, B, C])))


# %%
