from typing import Callable, List

import numpy as np

from net import SignPredictor
from activation_functions import der_relu, der_sigmoid, relu, sigmoid, softmax
from loss_functions import der_entropy_loss, entropy_loss
from optimizer import Optimizer


if __name__ == "__main__":
    optimizer = Optimizer(
        lr = 0.1,
        adam_args = {
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-8,
        },
        lr_decay_type= "cosine",
        total_epochs = 4000
        )
    net = SignPredictor(
        input_size = 4,
        layer_sizes= [3, 4],
        activations = [relu, softmax],
        activations_der=[der_relu, None],
        loss_fn = entropy_loss,
        loss_fn_der= None,
        optimizer=optimizer

    )
    np.matrix([
        [1,0,0,0],
        [0,1,0,0]
    ])
    inputs = [
        np.array([[1, 0, 0, 0]]).T,
        np.array([[0, 1, 0, 0]]).T,
        np.array([[0, 0, 1, 0]]).T,
        np.array([[0, 0, 0, 1]]).T,
    ]

    targets = [
        np.array([[1, 0, 0, 0]]).T,
        np.array([[0, 1, 0, 0]]).T,
        np.array([[0, 0, 1, 0]]).T,
        np.array([[0, 0, 0, 1]]).T,
    ]
    t = 1
    for epoch in range(1000):
        total_loss = 0
        for x, y in zip(inputs, targets):
            loss, pred = net.train_step(x, y, t)
            total_loss += loss
            t += 1
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}, learning rate: {optimizer.current_lr}")
        if epoch == 999:
            print(f"Final Epoch prediction: {pred.round()}, target: {y}")
