from transformers import AutoModelForImageClassification
import torch
from torch_mlir import fx


resnet18 = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
resnet18.eval()

def run(f):
    #print(f"{f.__name__}")
    #print("-" * len(f.__name__))
    f()
    print()

@run
def lower_pytorch_to_linalg_on_tensors():
    module = fx.export_and_import(resnet18, torch.ones(1, 3, 224, 224), output_type="torch", func_name="resnet18")

    with open("resnet18_model_torch.mlir", "w") as f:
        f.write(str(module))