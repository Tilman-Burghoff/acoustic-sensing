from abc import ABC, abstractmethod
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
        self.Xscaling = StandardScaler()
        self.yscaling = MinMaxScaler()

    def train(self, X, y):
        X_train = self.Xscaling.fit_transform(np.squeeze(X[:,:,0]))
        y_train = self.yscaling.fit_transform(y.reshape(-1, 1))
        untrained = self.MLP(X.shape[1], self.hidden_layers)
        self.model = train_nn(untrained, X_train, y_train)

    def predict(self, X):
        X_test = torch.tensor(self.Xscaling.transform(np.squeeze(X[:,:,0])), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            y_scaled = self.model(X_test).numpy()
        return self.yscaling.inverse_transform(y_scaled).ravel()

    class MLP(nn.Module):
        def __init__(self, inputdim, hidden_layers: int):
            super().__init__()
            if hidden_layers < 2:
                model = [nn.Linear(inputdim, 64), nn.ReLU()]
            else:
                model = [nn.Linear(inputdim, 512), nn.ReLU(), nn.Linear(512,64), nn.ReLU()]
                for _ in range(hidden_layers-2):
                    model.append(nn.Linear(64, 64))
                    model.append(nn.ReLU())
            model.append(nn.Linear(64, 1))
            self.linear_stack = nn.Sequential(*model)

        def forward(self, x):
            return self.linear_stack(x)
        

class Convolution(Model):
    def __init__(self, channels=1):
        super().__init__()
        self.channels = channels
        self.Xscaling = StandardScaler()
        self.yscaling = MinMaxScaler()

    def train(self, X, y):
        X = X[:,:,:self.channels]
        X_train = self.Xscaling.fit_transform(X.reshape(-1, X.shape[1])).reshape(-1, X.shape[1], self.channels)
        y_train = self.yscaling.fit_transform(y.reshape(-1, 1))
        untrained = self.Conv(self.channels)
        self.model = train_nn(untrained, X_train, y_train)

    def predict(self, X):
        X = X[:,:,:self.channels]
        X = self.Xscaling.transform(X.reshape(-1, X.shape[1])).reshape(-1, X.shape[1], self.channels)
        self.model.eval()
        with torch.no_grad():
            y_scaled = self.model(torch.tensor(X, dtype=torch.float32)).numpy()
        return self.yscaling.inverse_transform(y_scaled).ravel()
        


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
    optimizer = optim.SGD(network.parameters(), lr=0.005, momentum=0.9, weight_decay=0.01)

    # Training loop
    epochs = 200  # Number of epochs to train
    batch_size = 128  # Batch size for mini-batch gradient descent

    X_train_tensor = torch.tensor(X, dtype=torch.float32)
    y_train_tensor = torch.tensor(y, dtype=torch.float32)

    best_score = 1
    best_model = copy.deepcopy(network)


    for epoch in range(epochs):
        nanloss = 0
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
                nanloss += 1
        
        if nanloss > 0:
            print(f"Warning: encountered loss=NaN {nanloss} times in epoch {epoch}. That is {nanloss/(X_train_tensor.size(0)/batch_size):%} of the batches.")
            network = copy.deepcopy(best_model)
            optimizer = optim.SGD(network.parameters(), lr=0.005, momentum=0.9, weight_decay=0.01)
            continue
    
        network.eval()
        train_loss = criterion(network(X_train_tensor), y_train_tensor)

        if train_loss.item() < best_score:
            best_score = loss.item()
            best_model = copy.deepcopy(network)
            if best_score < 0.001:
                print(f"Abandoning training after {epoch} epochs with loss {best_score:.6f}")
                return best_model

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss.item():.6f}')
        
    return best_model