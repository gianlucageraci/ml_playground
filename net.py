from typing import Callable, List
import numpy as np 
from layers import LinearLayer
from optimizer import Optimizer
from loss_functions import binary_entropy_loss, entropy_loss, der_entropy_loss
from activation_functions import sigmoid, relu, der_relu, der_sigmoid, softmax

def create_network_architecture(
    input_size: int,
    layer_sizes: List[int],
    activations: List[Callable],
    activation_derivatives: List[Callable]
) -> List[LinearLayer]:
    """
    Constructs a list of LinearLayer objects defining a feedforward network.

    Args:
        input_size (int): Number of input features.
        layer_sizes (List[int]): List of output sizes for each layer.
        activations (List[Callable]): Activation functions for each layer.
        activation_derivatives (List[Callable]): Derivatives of the activation functions.

    Returns:
        List[LinearLayer]: Fully constructed network as a list of layers.
    """
    assert len(layer_sizes) == len(activations) == len(activation_derivatives), \
        "Mismatch between number of layers and activation functions"

    layers = []
    in_features = input_size

    for out_features, act, act_der in zip(layer_sizes, activations, activation_derivatives):
        layer = LinearLayer(
            in_features=in_features,
            out_features=out_features,
            act_func=act,
            act_func_der=act_der
        )
        layers.append(layer)
        in_features = out_features  # output of this layer is input to the next

    return layers
        
class SignPredictor:
    def __init__(self, 
                 input_size: int,
                 layer_sizes: List[int],
                 activations: List[callable],
                 activations_der: List[callable],
                 loss_fn, 
                 loss_fn_der, 
                 optimizer : Optimizer
                ):
        """Initialize the sign predictor model with a loss function.

        Args:
            loss_fn (callable, optional): Loss function for model training.
            loss_fn_der (callable, optional): Derivative of the loss function.
        """
        self.input_size = input_size
        self.loss_fn = loss_fn
        self.loss_fn_der = loss_fn_der
        self.optimizer = optimizer
        self.layers = create_network_architecture(input_size, layer_sizes, activations, activations_der)
  

    def forward(self, x):
        """Run a forward pass through the network.

        Constructs the network architecture with a hidden and output layer,
        and passes the input through each layer sequentially.

        Args:
            x (np.ndarray): Input array of shape (batch_sizte, input_features).

        Returns:
            np.ndarray: Model output from the final layer.
        """
        assert x.shape[1] == self.input_size, \
            f"The input passed to the net is of shape {x.shape} but is supposed to be of shape (batch_size, {self.input_size})"
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def loss_gradient(self, target):
        """
        Calculate the loss for the output layer (final layer)
        Needs to calculate: 
        dl/da (derivative of loss function at final output) x da/dz (derivative of activation function at last aggregate z)
        """
        output_layer = self.layers[-1]
        if self._check_loss_gradient_simplification(output_layer.act_func):            
            return output_layer.cache_dir["a_l"] - target
        else:
            return self.loss_fn_der(target, output_layer.cache_dir["a_l"]) * output_layer.act_func_der(output_layer.cache_dir["z_l"])

    def _check_loss_gradient_simplification(self, act_func):
        """
        If the final layer uses softmax as the activation function and entropy loss is used or (as the 2-dim simplification) 
        sigmoid is used with binary entropy loss, the derivative of loss function and output layer activation function can
        be simplified to pred - gt (y_hat - y)
        """
        return (self.loss_fn.__name__ == "entropy_loss" and act_func.__name__ == "softmax") or (self.loss_fn.__name__ == "binary_entropy_loss" and act_func.__name__ == "sigmoid")

    def backward(self, grad):
        """
        Implements the high-level gradient backpropagation by using a recursive loop and passing the
        calculations done in later layers to earlier ones, in order to reduce number of computations 
        Args:
            grad: The loss_gradient, i.e. the gradient calculated wrt to the output layer
        """
        for layer_idx in reversed(range(len(self.layers))):
            if layer_idx == len(self.layers) - 1:
                # For output layer, pass loss_gradient and None weights
                grad = self.layers[layer_idx].backpropagate(grad, None)
            else:
                # For hidden layers, pass grad and weights of next layer
                grad = self.layers[layer_idx].backpropagate(grad, self.layers[layer_idx + 1].weights)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def train_step(self, input_batch, target_batch, t:int):
        """Does a single training step with the received input"""
        # Forward pass
        pred = self.forward(input_batch)
        
        # Calculate loss
        loss = self.loss_fn(target_batch, pred)

        # Backpropagate
        loss_gradient = self.loss_gradient(target_batch)
        self.backward(loss_gradient)
        self.optimizer.step(t, self.layers)
        self.optimizer.zero_grad(self.layers)

        return loss, pred


