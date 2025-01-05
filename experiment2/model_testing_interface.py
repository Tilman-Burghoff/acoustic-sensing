from abc import ABC, abstractmethod
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch import optim
import copy

# Define an abstract class
class Model(ABC):
    @abstractmethod
    def train(self, X1, X2, X3, X4, y):
        pass


    @abstractmethod
    def predict(self, X1, X2, X3, X4,):
        pass


class Linear(Model):
    def __init__(self):
        super().__init__()
        self.reg = make_pipeline([StandardScaler, LinearRegression])

    def train(self, X1, X2, X3, X4, y):
        self.reg.fit(X1, y)

    def predict(self, X1, X2, X3, X4):
        return self.reg.predict(X1)


class FullyConnected(Model):
    def __init__(self, hidden_layers=1):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.scaling = StandardScaler()

    def train(self, X1, X2, X3, X4, y):
        X_train = self.scaling.fit_transform(X1)
        untrained = self.MLP(X1.shape[1], self.hidden_layers)
        self.model = train_nn(untrained, X_train, y)

    def predict(self, X1, X2, X3, X4,):
        X1 = torch.tensor(self.scaling.transform(X1), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            return self.model(X1).numpy().ravel()
        


    class MLP(nn.Module):
        def __init__(self, inputdim, hidden_layers: int):
            super().__init__()
            
            if hidden_layers == 0:
                self.model = [nn.Linear(inputdim, 64)]
            else:
                self.model = [nn.Linear(inputdim, 512)]
                self.model.append(nn.Linear(512,64))
                for i in range(hidden_layers-1):
                    self.model.append(nn.Linear(64, 64))
            self.model.append(nn.Linear(64, 1))
            self.relu = nn.ReLU()

        def forward(self, x):
            for layer in self.model[:-1]:
                x = self.relu(layer(x))
            return self.model[-1](x)
        

class Convolution(Model):
    def __init__(self, channels=1):
        super().__init__()
        self.channels = channels
        self.scaling = StandardScaler

    def train(self, X1, X2, X3, X4, y):
        input_X = [X1, X2, X3, X4]
        X = np.stack(input_X[:self.channels], axis=-1)
        X_train = self.scaling.fit_transform(X.reshape(-1, X1.shape[1])).reshape(-1, self.channels, X1.shape[1])
        untrained = self.Conv(self.channels)
        self.model = train_nn(untrained, X_train, y)

    def predict(self, X1, X2, X3, X4,):
        input_X = [X1, X2, X3, X4]
        X = np.stack(input_X[:self.channels], axis=-1)
        X = self.scaling.transform(X.reshape(-1, X1.shape[1])).reshape(-1, self.channels, X1.shape[1])
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(X)).numpy().ravel()
    
        


    class Conv(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv_relu_stack = nn.Sequential(
                nn.Conv1d(channels, 8, 16, 4),
                nn.ReLU(),
                nn.Conv1d(8, 16, 16, 4),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(960, 32),
                nn.ReLU(),
                nn.Linear(32,1)
            )

        def forward(self, x):
            x = torch.swapaxes(x, 1, 2)
            output = self.conv_relu_stack(x)
            return output




def train_nn(network, X, y):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01)

    # Training loop
    epochs = 1#00  # Number of epochs to train
    batch_size = 32  # Batch size for mini-batch gradient descent

    X_train_tensor = torch.tensor(X, dtype=torch.float32)
    y_train_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    best_score = 1
    losses = []

    for epoch in range(epochs):
        network.train()  # Set the model to training mode
        permutation = torch.randperm(X_train_tensor.size(0))  # Shuffle the data

        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

            # Zero the gradients from the previous step
            optimizer.zero_grad()

            # Forward pass
            outputs = network(batch_x)

            # Calculate the loss
            loss = criterion(outputs, batch_y)

            # Backward pass (compute gradients)
            loss.backward()

            # Update the weights using the optimizer
            optimizer.step()

        # Print the loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

        losses.append(loss.item())
        if loss.item() < best_score:
            best_score = loss.item()
            best_model = copy.deepcopy(network)
        
    return best_model