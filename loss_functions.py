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
    return - np.sum(gt * np.log(pred))


def binary_entropy_loss(gt, pred):
    """Calculate Cross Entropy loss
    an be used in the case that only a binary classification (e.g. 0 or 1) is implemented.
    Args:
        x (np.array): values predicted by neural network
        y (np.array): ground truth
    """
    m = pred.shape[1]
    loss = - (pred * np.log(gt) + (1-pred) * np.log(1-gt))
    empirical_loss = 1/m * np.nansum(loss)
    return empirical_loss


def der_entropy_loss(gt,pred):
    """Calculate Cross Entropy loss
    The x and y input should have the same shape and both represent a 
    probability distributions over class assignments.
    Args:
        x (np.array): values predicted by neural network of shape (out_features, 1)
        y (np.array): ground truth of shape (out_features, 1)
    """
    return pred - gt