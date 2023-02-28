import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class NoProphet:
    target_column = "y"
    time_column = "ds"

    model = LinearRegressionModel(1, 1)

    def fit(self, df: pd.DataFrame, learning_rate: float = 0.01, epochs: int = 10):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        ds: np.ndarray = df[self.time_column].values  # type: ignore
        y: np.ndarray = df[self.target_column].values  # type: ignore
        # print(ds, y)

        inputs = Variable(
            torch.from_numpy(
                np.array([ts for ts in range(len(ds))]).reshape(-1, 1)
            ).float()
        )
        labels = Variable(torch.from_numpy(y.reshape(-1, 1)).float())

        for epoch in range(epochs):
            optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            print(loss)

            loss.backward()

            optimizer.step()

            print("epoch {}, loss {}".format(epoch, loss.item()))

    def predict(self):
        pass
