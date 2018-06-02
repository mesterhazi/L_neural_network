import numpy as np
import matplotlib.pyplot as plt
import nn_activations
import random


class L_neural_network:
    def __init__(self, layer_dimensions, layer_activations, cost_function):
        self._initialize_parameters(layer_dimensions, layer_activations, cost_function)

    def _initialize_parameters(self, layer_dimensions, layer_activations, cost_function):
        """ Initializes the required parameter matrices
        Arguments:
            layer_dimensions: an array containing the number of nodes for each
                layer
            layer_activations: an array containing the activation functions and
                their derivatives in a tuple for every layer
        Sets: self.parameters containing the NN parameters W# and b# for each layer"""
        self.layer_dims = layer_dimensions
        self.layer_num = len(self.layer_dims)
        self.layer_activations = layer_activations
        self.parameters = {}
        self.cost_function = cost_function

        assert(len(self.layer_activations) == len(self.layer_dims),
        'Number of layers in layer_dimensions: {} and layer_activations: {} are not matching'.format(self.layer_num, len(self.layer_activations)))

        for l in range(1, self.layer_num):
            self.parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1])
            self.parameters['b' + str(l)] = np.zeros(self.layer_dims[l], 1)

    def _linear_forward(self, A, W, b):
        Z = W.dot(A) + b

        assert( Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)

        return Z, cache

    def _linear_activation_forward(self, A, W, b, activation):
        """ Arguments: activation: a function computing the activation of a layer (eg. sigmoid, ReLU)
                        activation should return A ,Z"""
        Z, linear_cache = self._linear_forward(A, W, b)
        A, activation_cache = activation(Z)

        cache = (linear_cache, activation_cache)
        return A, cache

    def _model_forward(self, X):
        """ Implements forward propagation for the whole network using the previously set layer sizes and activations
         Arguments:
             X: numpy array of the examples shape = (number of features, number of examples)
             parameters: dictionary with the parameters of the network Wi = parameters['Wi'], bi = parameters['bi'] for
             the i-th layer of the network
         Returns:
             AL: The predictions of the network
             caches: list of caches containing linear_activation_forwards cache:
                activation_cache: Z
                linear_cache: (A,W,b)
        """
        caches = []
        A = X

        for l in range(1, self.layer_num+1):
            A, cache = self._linear_activation_forward(
                A, self.parameters['W' + str(l)], self.parameters['b' + str(l)], self.layer_activations[l][0])
            caches.append(cache)

        assert (A.shape == (1, X.shape[1]))

        return A, caches

    def _compute_cost(self, AL, Y):
        return np.sqeeze(self.cost_function(AL, Y))

    def _linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1 / m * dZ.dot(A_prev.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = W.T.dot(dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db

    def _linear_activation_backward(self, dA, cache, activation_backward):
        """
        Implements one step of backward propagation using dA and the caches with the given activation function
        First the activation backward step is applied, then the linear backward step is completed
        :param dA:
        :param cache:
        :param activation_backward:
        :return:
        """
        linear_cache, activation_cache = cache
        dZ = activation_backward[1](dA, activation_cache)
        dA_prev, dW, db = self._linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def _model_backward(self, AL, Y, caches):
        grads = {}
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL


        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # derivative of cost with respect to AL
        dA, dW, db = self._linear_activation_backward(dAL,caches[-1],self.layer_activations[-1])
        grads["dA" + str(self.layer_num)], grads["dW" + str(self.layer_num)], grads["db" + str(self.layer_num)] = dA, dW, db

        for l in reversed(range(self.layer_num)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
            # print(l)
            current_cache = caches[l]
            # print(current_cache)
            # print(np.shape(dA_prev_temp))
            dA_prev_temp, dW_temp, db_temp = self._linear_activation_backward(grads["dA" + str(l + 2)], current_cache,
                                                                        self.layer_activations[l][1])
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
            ### END CODE HERE ###

        return grads

    def _update_parameters(self, grads, learning_rate):
        for l in range(self.layer_num):
            self.parameters["W" + str(l + 1)] = self.parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
            self.parameters["b" + str(l + 1)] = self.parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        return self.parameters

    def train_gradient_descent(self, X, Y, number_of_iterations, learning_rate, print_cost=False):
        costs = []
        for i in range(0, number_of_iterations):
            AL, caches = self._model_forward(X)

            # Compute cost.
            cost = self._compute_cost(AL, Y)

            # Backward propagation.
            grads = self._model_backward(AL, Y, caches)

            # Update parameters.
            parameters = self._update_parameters(grads, learning_rate)

            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)

            # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        return self.parameters

    def predict(self, X):
        return self._model_forward(X)


