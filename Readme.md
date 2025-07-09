# NN_from_scratch

A minimal neural network framework implemented from scratch in Python and NumPy, designed for educational purposes and experimentation.

## Features

- **Fully Connected Neural Networks**: Build and train networks with arbitrary layer sizes.
- **Custom Activation Functions**: Easily swap between ReLU, Sigmoid, and more.
- **Dropout Regularization**: Prevent overfitting by randomly dropping units during training.
- **Weight Decay (L2 Regularization)**: Penalize large weights to improve generalization.
- **Adam Optimizer**: Adaptive moment estimation for efficient and robust training.
- **Learning Rate Scheduling**: Cosine and exponential decay options for learning rate annealing.
- **Mini-batch Training**: Efficient training with support for mini-batch gradient descent.
- **Binary Cross-Entropy Loss**: Suitable for binary classification tasks.
- **Modular Design**: Easily extend with new layers, optimizers, or loss functions.

## Usage

See [`train_data.ipynb`](NN_from_scratch/train_data.ipynb) for an example of training a neural network on a binary image classification task.

## Project Structure

- [`net.py`](NN_from_scratch/net.py): Main network and training logic.
- [`optimizer.py`](NN_from_scratch/optimizer.py): Adam optimizer, weight decay, and learning rate scheduling.
- [`activation_functions.py`](NN_from_scratch/activation_functions.py): Activation functions and their derivatives.
- [`loss_functions.py`](NN_from_scratch/loss_functions.py): Loss functions for training.
- [`layers/`](NN_from_scratch/layers/): Layer implementations, including dropout.
- [`train_data.ipynb`](NN_from_scratch/train_data.ipynb): Example notebook for data loading and training.

## Getting Started

1. Install dependencies:
    ```sh
    pip install numpy scikit-learn
    ```
2. Run the example notebook:
    ```sh
    jupyter notebook NN_from_scratch/train_data.ipynb
    ```

---

**Author:** Gianluca  
**License:** MIT