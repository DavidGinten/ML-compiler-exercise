import torch

torch.manual_seed(41)

class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return self.linear(x + self.param).clamp(min=0.0, max=1.0)
    
if __name__ == "__main__":
    model = MyModule()
    out = model(torch.ones((3, 4)))
    print(' '.join(f'{x:.4f}' for x in out.view(-1)))
    #print("Output: ", model(x))