from typing import Callable
import numpy as np 

class LinearLayer:
    def __init__(self, 
            in_features: int, 
            out_features: int, 
            act_func: Callable, 
            act_func_der: Callable
        ):
        """Initialize a linear layer with an activation function.

        Args:
            in_features (int): Number of input features  (n_{l-1}).
            out_features (int): Number of output features (i.e., neurons: n_l).
            act_func (callable): Activation function to apply to the layer output.
            act_func_der (callable): Derivative of the activation function.
        """
        self._in_features = in_features
        self._out_features = out_features
        self.weights, self.bias = self._init_params()
        self.act_func = act_func
        self.act_func_der = act_func_der
        self.cache_dir = {
            "x": None,
            "a_l": None,
            "z_l": None
        }
        self.grad_l = None
        self.grad_w = None
        self.grad_b = None

        self.weights_m = np.zeros_like(self.weights)
        self.weights_s = np.zeros_like(self.weights)
        self.bias_m = np.zeros_like(self.bias)
        self.bias_s = np.zeros_like(self.bias)
            

    def _init_params(self, method = "random"):
        """Initialize weights and biases.

        Args:
            method (str): Initialization method. Default is 'random'.

        Returns:
            tuple: Initialized weights and biases as numpy arrays.
        """
        if method == "random":
            weights = np.random.randn(self._out_features, self._in_features) * np.sqrt(2.0 / self._in_features)
            bias    = np.zeros((1, self._out_features))
            return weights, bias

    def forward(self, x: np.ndarray):
        """Perform forward pass through the layer.

        Applies a linear transformation followed by the activation function:
            z_l = W_l * a_{l-1} + b_l  with a_{l-1} = x
            a_l = act_func(z_l)

        Args:
            x (np.ndarray): Input array of shape n_{l-1}x1.

        Returns:
            np.ndarray: Output array of shape n_lx1.
        """
        assert (self._in_features, 1) == x.shape, \
            f"Input is of shape {x.shape} but {self._in_features} input features were expected" # need to add batchsize
        
        self.cache_dir["x"] = x # a_{l-1}
        z_l = self.weights@x + self.bias #(out_features,1)
        self.cache_dir["z_l"] = z_l
        a_l = self.act_func(z_l) #(out_features,1)
        self.cache_dir["a_l"] = a_l
        return a_l
    
    def backpropagate(self, grad_next_layer, weights_next_layer):
        """Backpropagate gradients through the layer.

        Computes gradients with respect to the weights, biases, and input based on the chain rule 
        (meaning based on gradients calculated in previous layers):
            grad_l = W_{l+1}^T · grad_{l+1} * act_func_der(z_l)
            grad_w = grad_l · a_{l-1}^T
            grad_b = grad_l

        Args:
            grad_next_layer (np.ndarray): Gradient from the next layer, has shape n_{l+1} x 1
            weights_next_layer (np.ndarray): Weights of the next layer, has shape n_{l+1}, n_l

        Returns:
            tuple:
                - grad_l (np.ndarray): Gradient to propagate to the previous layer, has shape n_l x 1 
                - grad_w (np.ndarray): Gradient with respect to the layer weights, has shape n_l x n_{l-1}
                - grad_b (np.ndarray): Gradient with respect to the layer biases, has shape n_l x 1
        """
        if weights_next_layer is None:
            grad_l = grad_next_layer # Output Layer
        else:
            grad_l = weights_next_layer.T @ grad_next_layer * self.act_func_der(self.cache_dir["z_l"])
        self.grad_w = grad_l@self.cache_dir["x"].T
        self.grad_b = grad_l
        return grad_l
    
    def update_params(self, t:int, lr: float, beta_1, beta_2, epsilon):
        """Use ADAM to update layer parameters"""
        self.weights_m = beta_1 * self.weights_m + (1-beta_1) * self.grad_w
        self.weights_s = beta_2 * self.weights_s + (1-beta_2) * self.grad_w**2

        self.bias_m = beta_1 * self.bias_m + (1-beta_1) * self.grad_b
        self.bias_s = beta_2 * self.bias_s + (1-beta_2) * self.grad_b**2

        weights_m_hat = self.weights_m / (1-beta_1**t)
        weights_s_hat = self.weights_s / (1-beta_2**t)
        bias_m_hat = self.bias_m / (1-beta_1**t)
        bias_s_hat = self.bias_s / (1-beta_2**t)
        
        self.weights -= lr * weights_m_hat/(np.sqrt(weights_s_hat) + epsilon)
        self.bias -= lr * bias_m_hat/(np.sqrt(bias_s_hat) + epsilon)

    def zero_grad(self):
        self.grad_w = None #np.zeros_like(self.weights)
        self.grad_b = None #np.zeros_like(self.bias)