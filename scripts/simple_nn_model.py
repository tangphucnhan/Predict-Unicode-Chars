import numpy as np
import time
import cv2


class SimpleNeuralNetwork:
    def __init__(self, sizes, activation='sigmoid'):
        self.sizes = sizes

        if activation == 'relu':
            self.activation = self.relu
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
        else:
            raise ValueError("Activation support only 'relu' or 'sigmoid'.")

        self.params = self.initialize()
        # Save all intermediate values, i.e. activations
        self.cache = {}

    def relu(self, x, derivative=False):
        if derivative:
            x = np.where(x < 0, 0, x)
            x = np.where(x >= 0, 1, x)
            return x
        return np.maximum(0, x)

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
        return 1 / (1 + np.exp(-x))

    def tanh(self, x, derivative=False):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def softmax(self, x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)

    def initialize(self):
        # number of nodes in each layer
        input_layer = self.sizes[0]
        hidden_layer = self.sizes[1]
        output_layer = self.sizes[2]

        params = {
            "W1": np.random.randn(hidden_layer, input_layer) * np.sqrt(1. / input_layer),
            "b1": np.zeros((hidden_layer, 1), dtype="uint8") * np.sqrt(1. / input_layer),
            "W2": np.random.randn(output_layer, hidden_layer) * np.sqrt(1. / hidden_layer),
            "b2": np.zeros((output_layer, 1), dtype="uint8") * np.sqrt(1. / hidden_layer)
        }
        return params

    def initialize_optimizer(self):
        return {
            "W1": np.zeros(self.params["W1"].shape, dtype="uint8"),
            "b1": np.zeros(self.params["b1"].shape, dtype="uint8"),
            "W2": np.zeros(self.params["W2"].shape, dtype="uint8"),
            "b2": np.zeros(self.params["b2"].shape, dtype="uint8"),
        }

    def feed_forward(self, x):
        self.cache["X"] = x
        self.cache["Z1"] = np.matmul(self.params["W1"], self.cache["X"].T) + self.params["b1"]
        self.cache["A1"] = self.activation(self.cache["Z1"])
        self.cache["Z2"] = np.matmul(self.params["W2"], self.cache["A1"]) + self.params["b2"]
        self.cache["A2"] = self.softmax(self.cache["Z2"])
        return self.cache["A2"]

    def back_propagate(self, y, output):
        current_batch_size = y.shape[0]
        dZ2 = output - y.T
        dW2 = (1. / current_batch_size) * np.matmul(dZ2, self.cache["A1"].T)
        db2 = (1. / current_batch_size) * np.sum(dZ2, axis=1, keepdims=True)
        dA1 = np.matmul(self.params["W2"].T, dZ2)
        dZ1 = dA1 * self.activation(self.cache["Z1"], derivative=True)
        dW1 = (1. / current_batch_size) * np.matmul(dZ1, self.cache["X"])
        db1 = (1. / current_batch_size) * np.sum(dZ1, axis=1, keepdims=True)

        self.grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return self.grads

    def cross_entropy_loss(self, y, output):
        l_sum = np.sum(np.multiply(y.T, np.log(output)))
        m = y.shape[0]
        l = -(1. / m) * l_sum
        return l

    def optimize(self, l_rate=0.1, beta=.9):
        if self.optimizer == "sgd":
            for key in self.params:
                self.params[key] = self.params[key] - l_rate * self.grads[key]
        elif self.optimizer == "momentum":
            for key in self.params:
                self.momemtum_opt[key] = (beta * self.momemtum_opt[key] + (1. - beta) * self.grads[key])
                self.params[key] = self.params[key] - l_rate * self.momemtum_opt[key]
        else:
            raise ValueError("Optimizer is currently not support, please use 'sgd' or 'momentum' instead.")

    def accuracy(self, y, output):
        return np.mean(np.argmax(y, axis=-1) == np.argmax(output.T, axis=-1))

    def train(self, x_train, y_train, x_test, y_test, epochs=10,
              batch_size=64, optimizer='momentum', l_rate=0.1, beta=.9):
        self.epochs = epochs
        self.batch_size = batch_size
        num_batches = -(-x_train.shape[0] // self.batch_size)

        self.optimizer = optimizer
        if self.optimizer == 'momentum':
            self.momemtum_opt = self.initialize_optimizer()

        start_time = time.time()
        template = "Epoch {}: {:.2f}s, train acc={:.2f}, train loss={:.2f}, test acc={:.2f}, test loss={:.2f}"

        # Train
        for i in range(self.epochs):
            permutation = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]

            for j in range(num_batches):
                # Batch
                begin = j * self.batch_size
                end = min(begin + self.batch_size, x_train.shape[0] - 1)
                x = x_train_shuffled[begin:end]
                y = y_train_shuffled[begin:end]

                output = self.feed_forward(x)
                back_prop = self.back_propagate(y, output)
                self.optimize(l_rate=l_rate, beta=beta)

            # Evaluate performance
            output = self.feed_forward(x_train)
            train_acc = self.accuracy(y_train, output)
            train_loss = self.cross_entropy_loss(y_train, output)
            # Test data
            output = self.feed_forward(x_test)
            test_acc = self.accuracy(y_test, output)
            test_loss = self.cross_entropy_loss(y_test, output)
            print(template.format(i + 1, time.time() - start_time, train_acc, train_loss, test_acc, test_loss))

    def predict(self, img):
        img_h, img_w = img.shape
        img_c = 3
        thresh = len(img[img >= 2 * np.pi * (img_c ** 2)])
        if thresh >= img_h * img_w / 2:
            img = np.invert(img)
        a_where = np.argwhere(img >= 2 * np.pi * (img_c ** 2))
        meaning_x1 = max(0, min(a_where[:, 1]) - np.ndim(img))
        meaning_y1 = max(0, min(a_where[:, 0]) - np.ndim(img))
        meaning_x2 = min(img_w, max(a_where[:, 1]) + np.ndim(img))
        meaning_y2 = min(img_w, max(a_where[:, 0]) + np.ndim(img))
        mean = img[meaning_y1:meaning_y2, meaning_x1:meaning_x2]
        in_h, in_w = img_h - 4, img_w - 4
        if mean.shape[0] > in_h:
            mean = cv2.resize(mean, (int(in_h * mean.shape[1] / mean.shape[0]), in_h))
        if mean.shape[1] > in_w:
            mean = cv2.resize(mean, (in_w, int(in_w * mean.shape[0] / mean.shape[1])))
        black_piece = np.zeros((img_h, img_w), dtype="uint8")
        black_piece[img_h - mean.shape[0]:img_h, 0:mean.shape[1]] = mean
        img = np.reshape(black_piece, (1, black_piece.shape[0] * black_piece.shape[1]))
        return self.feed_forward(img)
