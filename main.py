import random
import numpy as np

from net import SignPredictor
from activation_functions import der_relu, relu, softmax
from loss_functions import entropy_loss
from optimizer import Optimizer


if __name__ == "__main__":
    optimizer = Optimizer(
        lr = 0.01,
        adam_args = {
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-8,
        },
        lr_decay_type= "cosine",
        total_epochs = 1000
        )
    # Generate inputs:
    inputs = [] #np.array([0.5, 0.5]).T]
    targets = [] #np.array([0.5, 0.5]).T]

    for i in range (20):
        val = round(random.random(), 1)
        if val > 0.5:
            targets.append(np.array([1,0]))
        elif val == 0.5:
            continue #targets.append(np.array([0.5, 0.5]))
        else:
            targets.append(np.array([0,1]))
        inputs.append(np.array([val, 1-val]))

    inputs = np.array(inputs)
    targets = np.array(targets)

    net = SignPredictor(
        input_size = inputs.shape[1],
        layer_sizes= [4, 2],
        activations = [relu, softmax],
        activations_der=[der_relu, None],
        loss_fn = entropy_loss,
        loss_fn_der= None,
        optimizer=optimizer
    )
    t = 1
    for epoch in range(1000):
        loss, pred = net.train_step(inputs, targets, t)
        t += 1
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}, learning rate: {optimizer.current_lr}")
        if epoch == 999:
            print(f"Final Epoch prediction: {pred[0].round()}, target: {targets[0]}, input:{inputs[0]}")