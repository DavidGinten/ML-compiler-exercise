import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(41)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
if __name__ == "__main__":
    model = NeuralNetwork()
    out = model(torch.ones((1, 28, 28)))
    print(' '.join(f'{x:.4f}' for x in out.view(-1)))
    #print("Input: ", x)
    #print("Output: ", model(x))