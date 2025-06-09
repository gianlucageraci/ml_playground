"""Contains all activation functions and there derivatives"""
import numpy as np

# For now, x is a single value 

def linear(x: np.ndarray):
    """Linear Activation Function"""
    return x

def der_linear(x: np.ndarray):
    """Derivitaive of linear function
    Args:
        x (np.ndarray): x should be a n-dim array
    Returns:
        np.ndarray: n-dim transformed array 
    """
    return np.ones_like(x)


def sigmoid(x: np.ndarray):
    """Sigmoid Activation Function
    Args:
        x (np.ndarray): x should be a n-dim array
    Returns:
        np.ndarray: n-dim transformed array 
    """
    return 1/(1+np.exp(-x))

def der_sigmoid(x:np.ndarray):
    """Derivative of Sigmoid Activation Function
     Args:
        x (np.ndarray): x should be a n-dim array
    Returns:
        np.ndarray: n-dim transformed array 
    """
    return sigmoid(x) * (1-sigmoid(x))


def tanh(x: np.ndarray):
    """Tanh Activation Function
    Args:
        x (np.ndarray): x should be a n-dim array
    Returns:
        np.ndarray: n-dim transformed array 
    """
    return np.tanh(x)

def der_tanh(x):
    """
    Derivative of Tanh Activation Function
    Args:
        x (np.ndarray): x should be a n-dim array
    Returns:
        np.ndarray: n-dim transformed array 
    """
    return 1- tanh(x)**2


def softmax(x: np.ndarray):
    """Stable softmax for vector inputs."""
    assert isinstance(x, np.ndarray), f"x must be a numpy array but is {type(x)}"
    shift = np.max(x, axis=1, keepdims=True)      # shape (batch_size, 1)
    e_x = np.exp(x - shift)                        # shape (batch_size, n_l)
    sum_e = np.sum(e_x, axis=1, keepdims=True)     # shape (batch_size, 1)
    return e_x / sum_e                             # broadcasting to (batch_size, n_l)

def der_softmax(x: np.ndarray):
    """
    ReLU Activation Function
    Args:
        x (np.ndarray): x should be a n-dim array
    Returns:
        np.ndarray: n-dim transformed array 
    """
    return np.diagflat(x) - np.outer(x, x)


def relu(x: np.ndarray):
    """
    ReLU Activation Function
    Args:
        x (np.ndarray): x should be a n-dim array
    Returns:
        np.ndarray: n-dim transformed array 
    """
    return np.maximum(x, 0)

def der_relu(x: np.ndarray):
    """Derivative of ReLU Activation Function
    Args:
        x (np.ndarray): x should be a n-dim array
    Returns:
        np.ndarray: n-dim transformed array 
    """
    return np.where(x <= 0, 0, np.where(x > 0, 1, x))


def leaky_relu(x: np.ndarray, leaky = False):
    """
    ReLU Activation Function
    Args:
        x (np.ndarray): x should be a n-dim array
    Returns:
        np.ndarray: n-dim transformed array 
    """
    return np.maximum(x*0.01, x)

def der_relu(x: np.ndarray, leaky = False):
    """Derivative of ReLU Activation Function
    Args:
        x (np.ndarray): x should be a n-dim array
    Returns:
        np.ndarray: n-dim transformed array 
    """
    return np.where(x <= 0, 0.01, np.where(x > 0, 1, x))
