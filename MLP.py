import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, activation_fn):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            activation_fn,
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x):
        return self.model(x)

def get_activation(name):
    return {
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "relu": nn.ReLU()
    }[name]

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)


def trainingLoop(num_of_epochs, hidden_channels, activation_fn, train_data,test_data, learning_rate, momentum):
    train_losses = []
    test_losses = []
    loss_fn = nn.MSELoss()
    activation_fn = get_activation(activation_fn)
    model = MLP(in_channels=2,
                hidden_channels=hidden_channels,
                out_channels=2,
                activation_fn=activation_fn)
    initialize_weights(model)

    size_of_train = len(train_data)
    size_of_test = len(test_data)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=size_of_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=size_of_test, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_of_epochs):
        model.train()
        epoch_train_losses = []

        for features, targets in train_loader:
            outputs = model(features)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_test_losses = []
        with torch.no_grad():
            for features, targets in test_loader:
                outputs = model(features)
                loss = loss_fn(outputs, targets)
                epoch_test_losses.append(loss.item())


        avg_test_loss = np.mean(epoch_test_losses)
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch + 1}/{num_of_epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")

    return model, train_losses, test_losses