import argparse
from MLP import trainingLoop
from data_preparation import loading_all_files
import numpy as np
from torch.utils.data import TensorDataset
import torch

def to_tensor(data):
    if hasattr(data, 'to_numpy'):
        return torch.tensor(data.to_numpy(), dtype=torch.float32)
    else:
        return torch.tensor(data, dtype=torch.float32)

def compute_cdf_error(x_measured, y_measured, x_true, y_true):
    distances = np.sqrt((x_measured - x_true) ** 2 + (y_measured - y_true) ** 2)
    sorted_distances = np.sort(distances)
    cdf = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
    return sorted_distances, cdf

def main(args):
    train_input, train_output, test_input, test_output = loading_all_files()

    train_data = TensorDataset(to_tensor(train_input), to_tensor(train_output))

    test_data = TensorDataset(to_tensor(test_input), to_tensor(test_output))

    model, train_loss, test_loss = trainingLoop(
        num_of_epochs=args.epochs,
        hidden_channels=args.hidden_size,
        activation_fn=args.activation,
        train_data=train_data,
        test_data=test_data,
        learning_rate=args.learning_rate,
        momentum=args.momentum
    )

    predictions_scaled = model(to_tensor(test_input)).detach().numpy()
    return model, train_loss, test_loss, predictions_scaled

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trenuj MLP z wybranymi parametrami.")
    parser.add_argument("--hidden_size", type=int, default=64, help="Liczba neuronów w warstwie ukrytej")
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh"], default="relu", help="Funkcja aktywacji")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Współczynnik uczenia")
    parser.add_argument("--momentum", type=float, default=0.9, help="Wartość momentum")
    parser.add_argument("--epochs", type=int, default=100, help="Liczba epok")

    args = parser.parse_args()
    main(args)
