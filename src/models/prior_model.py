import torch
import torch.nn as nn

class PriorModel(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer(
            "logits",
            torch.log(torch.ones(num_classes) / num_classes)
        )


    def fit(self, y: torch.Tensor):

        counts = torch.bincount(y)
        probs = counts / counts.sum()
        self.logits.data = torch.log(probs.to(self.logits.device))


    def forward(self, X: torch.Tensor):

        return self.logits.unsqueeze(0).expand(X.shape[0], self.num_classes)
    

    def predict(self, X):

        logits = self.forward(X)
        return torch.argmax(logits, dim=1)
    