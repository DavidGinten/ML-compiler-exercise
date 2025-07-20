import torch, time
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(41)

class ConvolutionalNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1,6,3,1)
    self.conv2 = nn.Conv2d(6,16,3,1)
    # Fully Connected Layer
    self.fc1 = nn.Linear(5*5*16, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, X):
    X = F.relu(self.conv1(X))
    X = F.max_pool2d(X,2,2) # 2x2 kernal and stride 2
    # Second Pass
    X = F.relu(self.conv2(X))
    X = F.max_pool2d(X,2,2) # 2x2 kernal and stride 2

    # Re-View to flatten it out
    X = X.view(-1, 16*5*5) # negative one so that we can vary the batch size

    # Fully Connected Layers
    X = F.relu(self.fc1(X))
    X = F.relu(self.fc2(X))
    X = self.fc3(X)
    return F.log_softmax(X, dim=1)
    

def print_output():
    model = ConvolutionalNetwork()
    out = model(torch.ones((32, 1, 28, 28)))
    print(' '.join(f'{x:.4f}' for x in out.view(-1)))
    #print("Input: ", x)
    #print("Output: ", model(x))

def benchmark():
    model = ConvolutionalNetwork()#.eval()
    x = torch.ones((32, 1, 28, 28))
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