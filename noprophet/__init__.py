from sched import scheduler
import matplotlib.dates as plt_dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable


def to_indices(ds: np.ndarray):
    return np.array([i for i in range(len(ds))])


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class NoProphet:
    target_column = "y"
    time_column = "ds"

    def __init__(self):
        self.model = LinearRegressionModel(1, 1)

    def fit(self, df: pd.DataFrame, learning_rate: float = 0.01, epochs: int = 10):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 0.0001)

        ds: np.ndarray = df[self.time_column].values  # type: ignore
        y: np.ndarray = df[self.target_column].values  # type: ignore
        # print(ds, y)

        inputs = Variable(torch.from_numpy(to_indices(ds).reshape(-1, 1)).float())
        labels = Variable(torch.from_numpy(y.reshape(-1, 1)).float())

        for epoch in range(epochs):
            optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            print("epoch {}, loss {}".format(epoch, loss.item()))

            scheduler.step()

        return ds, y

    def predict(self, df: pd.DataFrame):
        with torch.no_grad():
            ds: np.ndarray = df[self.time_column].values  # type: ignore
            inputs = Variable(torch.from_numpy(to_indices(ds).reshape(-1, 1)).float())
            forecast = self.model(inputs).data.numpy()
            return forecast.reshape(-1)

    def plot(self, ds: np.ndarray, y: np.ndarray, forecast: np.ndarray):
        plt.clf()
        plt.plot(ds, y, "-", label="Values (y)", alpha=0.5)
        plt.plot(ds, forecast, "-", label="Prediction (yhat)", alpha=0.5)
        plt.legend(loc="best")
        plt.gca().xaxis.set_major_locator(plt_dates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        plt.show()
