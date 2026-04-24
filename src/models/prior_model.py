import torch
import torch.nn as nn

class PriorModel(nn.Module):

    def __init__(self, num_classes):

        self.num_classes = num_classes
        self.logits = torch.log(torch.ones(num_classes) / num_classes)


    def fit(self, y: torch.Tensor):

        counts = torch.bincount(y)
        probs = counts / counts.sum()
        self.logits = torch.log(probs)


    def forward(self, X: torch.Tensor):

        return self.logits.unsqueeze(0).expand(X.shape[0], self.num_classes)
    

    def predict(self, X):

        logits = self.forward(X)
        return torch.argmax(logits, dim=1)
    
    def to(self, device):
        self.logits = self.logits.to(device)
        return self
    
    
    def __call__(self, X):
        return self.forward(X)