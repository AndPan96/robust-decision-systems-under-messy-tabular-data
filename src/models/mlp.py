import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, input_dim, num_classes):
    
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, num_classes)
        )


    def forward(self, X):

        return self.net(X)

