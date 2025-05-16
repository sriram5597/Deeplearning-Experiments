import numpy as np
from .activation import activation, activation_gradient

np.random.seed(3)


class FeedForwardNetwork:
    def __init__(self, L, layer_dims, activations) -> None:
        self.L = L
        self.layer_dims = layer_dims
        self.activations = activations
        self.cache = dict()
        self.parameters = {}
        self.grads = {}
        self.m = 0

    def _initialize_parameters(self):
        parameters = [
            {},
        ]
        if self.L == 1:
            parameters.append(
                {
                    "W": np.random.randn(self.layer_dims[1], self.layer_dims[0]) * 0.01,
                    "b": np.zeros((self.layer_dims[1], self.layer_dims[0])),
                }
            )
            return parameters
        for i in range(1, self.L + 1):
            w = np.random.randn(self.layer_dims[i], self.layer_dims[i - 1]) * 0.01
            b = np.zeros((self.layer_dims[i], 1))
            assert w.shape == (self.layer_dims[i], self.layer_dims[i - 1])
            parameters.append({"W": w, "b": b})
        self.parameters = parameters

    def _forward(self, layer):
        parameter = self.parameters[layer]
        prev_A = self.cache[f"{layer - 1}"]["A"]
        Z = np.dot(parameter["W"], prev_A) + parameter["b"]
        activation_fn = self.activations[layer - 1]
        A = activation(Z, activation_fn)
        self.cache[f"{layer}"]["A"] = A
        self.cache[f"{layer}"]["Z"] = Z

    def _forward_prop(self, X):
        self.cache["0"] = {"A": X}
        for i in range(1, self.L + 1):
            self.cache[str(i)] = {}
            self._forward(i)
        return self.cache[f"{self.L}"]["A"]

    def _cost(self, Y, Y_hat):
        m = Y.shape[1]
        cost = (
            -1
            / m
            * np.sum(
                Y * np.log(np.clip(Y_hat, 1e-15, 1 - 1e-15))
                + (1 - Y) * np.log(np.clip(1 - Y_hat, 1e-15, 1 - 1e-15))
            )
        )
        return np.squeeze(cost)

    def _backward_prop(self, Y, Y_hat):
        self.grads = {
            f"{self.L}": {"dA": -(np.divide(Y, Y_hat) - np.divide((1 - Y), (1 - Y_hat)))}
        }

        for i in range(self.L, 0, -1):
            grad = self.grads[f"{i}"]
            cache = self.cache[f"{i}"]
            parameter = self.parameters[i]
            prev_cache = self.cache[f"{i - 1}"]
            dA = grad["dA"]

            dZ = np.multiply(
                dA, activation_gradient(cache["Z"], self.activations[i - 1])
            )
            dW = np.dot(dZ, prev_cache["A"].T) / self.m
            dB = np.sum(dZ, axis=1, keepdims=True) / self.m
            dA_prev = np.dot(parameter["W"].T, dZ)

            self.grads[str(i)]["dW"] = dW
            self.grads[str(i)]["dB"] = dB
            self.grads[str(i - 1)] = {"dA": dA_prev}

    def _update_parameters(self, learning_rate):
        for l in range(1, self.L + 1):
            gradient = self.grads[str(l)]
            self.parameters[l]["W"] = (
                self.parameters[l]["W"] - learning_rate * gradient["dW"]
            )
            self.parameters[l]["b"] = (
                self.parameters[l]["b"] - learning_rate * gradient["dB"]
            )

    def train(self, X, Y, learning_rate=0.1, iterations=500):
        cost = 0
        self._initialize_parameters()
        for i in range(iterations):
            Y_hat = self._forward_prop(X)
            self.m = Y.shape[1]
            cost = self._cost(Y, Y_hat)
            self._backward_prop(Y, Y_hat)
            self._update_parameters(learning_rate)
            if i % 100 == 0:
                print(f"cost after {i} iterations: ", cost)
        return cost

    def predict(self, X):
        Y_hat = self._forward_prop(X)
        return Y_hat

    def info(self):
        print("Layers: ", self.L)
        print("Layer Dims: ", self.layer_dims)
