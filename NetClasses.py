# Imports

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model

class ShallowNetwork:
    """
    Class to create a shallow neural network

    Parameters:
    n_inputs: Number of inputs to the network
    n_hidden: Number of hidden units in the network
    n_outputs: Number of outputs from the network
    activation: Activation function to use
    optimizer: Optimizer to use
    """
    

    def __init__(self, n_inputs, n_hidden, n_outputs, activation=tf.nn.relu, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.activation = activation
        self.optimizer = optimizer
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(self.n_hidden, activation=self.activation, input_shape=(self.n_inputs,)))
        model.add(tf.keras.layers.Dense(self.n_outputs))
        return model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_weights(self):
        return self.model.get_weights()
    
    def train(self, ds, loss=None, epochs=100, batch_size=100):
        if loss:
            self.model.compile(loss=loss, optimizer=self.optimizer)
        else:
            self.model.compile(loss='mse', optimizer='adam')

        history = self.model.fit(ds, epochs=epochs, verbose=0, batch_size=batch_size)
        return history


class FunctionSamples:

    def __init__(self, function, n_samples, n_inputs, n_outputs):
        self.function = function
        self.n_samples = n_samples
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    def __call__(self):

        X = tf.random.uniform((self.n_samples, self.n_inputs), -1, 1)
        Y = tf.map_fn(lambda x: self.function(*x), X)
        yield X, Y

    def get_dataset(self):
        return tf.data.Dataset.from_generator(
            self, (tf.float32, tf.float32), ((self.n_samples, self.n_inputs), (self.n_samples,))
            )
    
    def hist(self):
        X, Y = next(self())

        return plt.hist(Y)

class BinaryNetwork:
    def __init__(self, tree_depth, hidden_units = 10):
        self.tree_depth = tree_depth
        self.model = self.build_model()
        self.hidden_units = hidden_units

    def binary_tree_nn(self, inputs, current_depth=0,hidden_units = 10):
        if current_depth == self.tree_depth:
            return Dense(hidden_units, activation='sigmoid')(inputs)

        left_branch = self.binary_tree_nn(inputs, current_depth + 1)
        right_branch = self.binary_tree_nn(inputs, current_depth + 1)

        concatenated = Concatenate()([left_branch, right_branch])
        # return Dense(2**(self.tree_depth - current_depth - 1), activation='relu')(concatenated)
        return Dense(2**(self.tree_depth - current_depth - 1)*hidden_units, activation='relu')(concatenated)

    def build_model(self):
        input_layer = Input(shape=(2**self.tree_depth,))
        output_layer = self.binary_tree_nn(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def summary(self):
        self.model.summary()

    def diagram(self):
        return tf.keras.utils.plot_model(self.model, show_shapes=True)



class DAGNetwork:
    def __init__(self, graph, input_dim, neurons_per_node, output_dim):
        self.graph = graph  # the graph dict: node -> [nodes it points to]
        self.input_dim = input_dim
        self.neurons_per_node = neurons_per_node
        self.output_dim = output_dim
        self.model = self.build_model()

    def build_model(self):
        inputs = Input(shape=(self.input_dim,))
        layer_dict = {}

        # Create a layer for each node in the graph
        for node in self.graph:
            layer_dict[node] = Dense(self.neurons_per_node, activation='relu')

        # Connect layers according to the graph
        output_layers = []
        for node, children in self.graph.items():
            if children:
                for child in children:
                    if child not in layer_dict:
                        raise ValueError(f"Node {child} not defined in graph")
                    # connect node output to each child's layer
                    x = layer_dict[node](inputs if node == 'input' else layer_dict[node].output)
                    layer_dict[child].output = Concatenate()([layer_dict[child].output, x]) if hasattr(layer_dict[child], 'output') else x
            else:
                # Collect outputs for nodes that don't point to others
                output_layers.append(layer_dict[node](inputs if node == 'input' else layer_dict[node].output))

        # If there are multiple outputs, concatenate them
        if len(output_layers) > 1:
            concatenated_outputs = Concatenate()(output_layers)
            final_output = Dense(self.output_dim, activation='softmax')(concatenated_outputs)
        else:
            final_output = Dense(self.output_dim, activation='softmax')(output_layers[0])

        model = Model(inputs=inputs, outputs=final_output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def summary(self):
        self.model.summary()

    def diagram(self):
        return tf.keras.utils.plot_model(self.model, show_shapes=True, show_layer_names=True)
