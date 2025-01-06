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
    def train(self, X, y):
        pass


    @abstractmethod
    def predict(self, X):
        pass


class Linear(Model):
    def __init__(self):
        super().__init__()
        self.scaling = StandardScaler()
        self.regression = LinearRegression()

    def train(self, X, y):
        X_train = self.scaling.fit_transform(np.squeeze(X[:,:,0]))
        self.regression.fit(X_train, y)

    def predict(self, X):
        X_test = self.scaling.transform(np.squeeze(X[:,:,0]))
        return self.regression.predict(X_test)


class FullyConnected(Model):
    def __init__(self, layers=2):
        super().__init__()
        self.hidden_layers = layers - 1
        self.scaling = StandardScaler()

    def train(self, X, y):
        X_train = self.scaling.fit_transform(np.squeeze(X[:,:,0]))
        untrained = self.MLP(X.shape[1], self.hidden_layers)
        self.model = train_nn(untrained, X_train, y)

    def predict(self, X):
        X_test = torch.tensor(self.scaling.transform(np.squeeze(X[:,:,0])), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            return self.model(X_test).numpy().ravel()
        


    class MLP(nn.Module):
        def __init__(self, inputdim, hidden_layers: int):
            super().__init__()
            
            if hidden_layers < 2:
                model = [nn.Linear(inputdim, 64)]
            else:
                model = [nn.Linear(inputdim, 512)]
                model.append(nn.Linear(512,64))
                for i in range(hidden_layers-1):
                    model.append(nn.Linear(64, 64))
            model.append(nn.Linear(64, 1))
            self.model = nn.ModuleList(model)
            self.relu = nn.ReLU()

        def forward(self, x):
            for layer in self.model[:-1]:
                x = self.relu(layer(x))
            return self.model[-1](x)
        

class Convolution(Model):
    def __init__(self, channels=1):
        super().__init__()
        self.channels = channels
        self.scaling = StandardScaler()

    def train(self, X, y):
        X = X[:,:,:self.channels]
        X_train = self.scaling.fit_transform(X.reshape(-1, X.shape[1])).reshape(-1, X.shape[1], self.channels)
        untrained = self.Conv(self.channels)
        self.model = train_nn(untrained, X_train, y)

    def predict(self, X):
        X = X[:,:,:self.channels]
        X = self.scaling.transform(X.reshape(-1, X.shape[1])).reshape(-1, X.shape[1], self.channels)
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(X, dtype=torch.float32)).numpy().ravel()
    
        


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
    epochs = 100  # Number of epochs to train
    batch_size = 64  # Batch size for mini-batch gradient descent

    X_train_tensor = torch.tensor(X, dtype=torch.float32)
    y_train_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    best_score = 1
    best_model = network

    for epoch in range(epochs):
        network.train()  # Set the model to training mode
        permutation = torch.randperm(X_train_tensor.size(0))  # Shuffle the data
        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

            optimizer.zero_grad()
            outputs = network(batch_x)

            loss = criterion(outputs, batch_y)

            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
            else:
                print(f"Encoutered NaN-loss in epoch {epoch} at index {i}")
        

        train_loss = criterion(network(X_train_tensor), y_train_tensor)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss.item():.6f}')



        if train_loss.item() < best_score:
            best_score = loss.item()
            best_model = copy.deepcopy(network)
        
    return best_model

