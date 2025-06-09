import numpy as np

def entropy_loss(gt: np.ndarray, pred: np.ndarray):
    """Calculate Cross Entropy loss
    The x and y input should have the same shape and both represent a 
    probability distributions over class assignments.
    Args:
        x (np.array): values predicted by neural network of shape (output_size, 1)
        y (np.array): ground truth (output_size, 1)
    """
    assert pred.shape == gt.shape, \
        f"Pred and ground truth are supposed to have the same shape, but are {pred.shape} and {gt.shape}"
    eps = 1e-15
    pred = np.clip(pred, eps, 1 - eps)
    return -np.mean(np.sum(gt * np.log(pred), axis=1))


def binary_entropy_loss(gt, pred):
    """Calculate Cross Entropy loss
    an be used in the case that only a binary classification (e.g. 0 or 1) is implemented.
    Args:
        x (np.array): values predicted by neural network of shape (batch_size, 1)
        y (np.array): ground truth (batch_size, 1)
    """
    epsilon = 1e-12
    pred = np.clip(pred, epsilon, 1 - epsilon)
    loss = - (gt * np.log(pred) + (1 - gt) * np.log(1 - pred))
    return np.mean(loss)


def der_entropy_loss(gt,pred):
    """Calculate Cross Entropy loss
    The x and y input should have the same shape and both represent a 
    probability distributions over class assignments.
    Args:
        x (np.array): values predicted by neural network of shape (out_features, 1)
        y (np.array): ground truth of shape (out_features, 1)
    """
    return pred - gt


def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))