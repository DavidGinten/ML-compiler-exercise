import torch, time
import torch.nn as nn
import torch.nn.functional as F

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

torch.manual_seed(41)

class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return self.linear(x + self.param).clamp(min=0.0, max=1.0)
    

def print_output():
    model = MyModule()
    out = model(torch.ones((3, 4)))
    print(' '.join(f'{x:.4f}' for x in out.view(-1)))
    #print("Input: ", x)
    #print("Output: ", model(x))

def benchmark():
    model = MyModule()#.eval()
    x = torch.ones((3, 4))
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
   print_output()
   #benchmark()