import torch, time
import torch.nn as nn
import torch.nn.functional as F

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
    

def print_output():
    model = NeuralNetwork()
    out = model(torch.ones((1, 28, 28)))
    print(' '.join(f'{x:.4f}' for x in out.view(-1)))
    #print("Input: ", x)
    #print("Output: ", model(x))

def benchmark():
    model = NeuralNetwork()#.eval()
    x = torch.ones((1, 28, 28))
    #with torch.no_grad():
    # Warm-up
    for _ in range(10):
        _ = model(x)

    start = time.time()
    for _ in range(100):
        _ = model(x)
    end = time.time()

    print(f"Avg inference time: {(end - start) / 100:.6f} sec")



if __name__ == "__main__":
   #print_output()
   benchmark()