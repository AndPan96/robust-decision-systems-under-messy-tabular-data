import torch.nn as nn

class LinearModel(nn.Module):

    def __init__(self, input_dim, num_classes):

        super().__init__()

        self.linear = nn.Linear(input_dim, num_classes)


    def forward(self, X):

        return self.linear(X)