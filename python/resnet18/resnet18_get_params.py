from transformers import AutoImageProcessor, ResNetForImageClassification, AutoModelForImageClassification
import torch
from datasets import load_dataset
import torchvision
from torch_mlir import torchscript
from torch_mlir import fx
#import torch_mlir
from torch_mlir.compiler_utils import (
    OutputType,
    run_pipeline_with_repro_report,
)

#dataset = load_dataset("huggingface/cats-image")
#image = dataset["test"]["image"][0]

#processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

# model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
resnet18 = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
#resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18.eval()

def run(f):
    #print(f"{f.__name__}")
    #print("-" * len(f.__name__))
    f()
    print()

@run
def lower_pytorch_to_linalg_on_tensors():
    import torch.utils._pytree as pytree
    import struct
    params = {
        **dict(resnet18.named_buffers(remove_duplicate=False)),
    }
    params_flat, params_spec = pytree.tree_flatten(params)
    params_flat = list(params_flat)
    #print(params_flat)

    def tensor_to_hex(tensor):
        """Convert a tensor of float32 values to concatenated hex representation."""
        # Ensure tensor is float32
        tensor = tensor.float()
        
        # Convert to numpy for easier byte manipulation
        numpy_array = tensor.detach().cpu().numpy()
        
        # Convert each float32 to 4 bytes, then to hex
        hex_parts = []
        for value in numpy_array:
            # Pack as little-endian float32 (4 bytes)
            bytes_repr = struct.pack('<f', value)
            # Convert bytes to hex string (without '0x' prefix)
            hex_str = bytes_repr.hex()
            hex_parts.append(hex_str)
        
        # Concatenate all hex strings and add '0x' prefix
        concatenated_hex = '0x04000000' + ''.join(hex_parts).upper()
        return concatenated_hex

    with open("params.txt", "w") as f:
        nums = []
        for i, tensor in enumerate(params_flat):
            if tensor.shape == torch.Size([]):
                continue
            hex_result = tensor_to_hex(tensor)
            f.write(f"torch_tensor_custom_{i}.float32: \"" + hex_result + "\",\n")
            nums.append((i, (len(hex_result)-10)//8))
        
        for (i, n) in nums:
            f.write(f"%arg{i} = torch.vtensor.literal(dense_resource<torch_tensor_custom_{i}.float32> : tensor<{n}xf32>) : !torch.vtensor<[{n}],f32>\n")