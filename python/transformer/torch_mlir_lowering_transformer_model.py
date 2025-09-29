from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

from torch.export import Dim
from torch_mlir import fx

"""from torch_mlir.compiler_utils import (
    OutputType,
    run_pipeline_with_repro_report,
)"""

# 1. CREATE STATIC SHAPES
def create_static_inputs(tokenizer, max_length=128, batch_size=1):
    """Create inputs with fixed, static shapes."""
    # Use dummy sentence to get consistent shapes
    dummy_sentences = ["This is a sample sentence for shape consistency."] * batch_size
    
    encoded_input = tokenizer(
        dummy_sentences,
        padding='max_length',          # Force max padding
        max_length=max_length,         # Fixed sequence length
        truncation=True,
        return_tensors='pt'
    )
    
    # Verify all shapes are static
    print("Input shapes (should be static):")
    for k, v in encoded_input.items():
        print(f"  {k}: {v.shape}")
    
    return encoded_input

# 2. IMPROVED WRAPPER
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        # Ensure inputs are the right type
        input_ids = input_ids.long()
        token_type_ids = token_type_ids.long()
        attention_mask = attention_mask.long()
        
        outputs = self.model(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state


# 3. EXPORT WITH CONSTRAINTS
def export_to_mlir(model, tokenizer, max_length=128, batch_size=1):
    """Export model to MLIR with proper shape handling."""
    
    # Create static inputs
    encoded_input = create_static_inputs(tokenizer, max_length, batch_size)
    #wrapped_model = Wrapper(model)
    """
    # Option A: Static shapes (recommended for avoiding -1 issues)
    try:
        ep = torch.export.export(
            wrapped_model,
            (
                encoded_input["input_ids"],
                encoded_input["token_type_ids"], 
                encoded_input["attention_mask"],
            )
        )
        print("✓ Static export successful")
        
    except Exception as e:
        print(f"Static export failed: {e}")
        print("Trying with dynamic shapes...")
        
        # Option B: Dynamic shapes with constraints
        batch_dim = Dim("batch", min=1, max=32)
        seq_dim = Dim("seq", min=1, max=512)
        
        dynamic_shapes = {
            "input_ids": {0: batch_dim, 1: seq_dim},
            "token_type_ids": {0: batch_dim, 1: seq_dim},
            "attention_mask": {0: batch_dim, 1: seq_dim},
        }
    """    
    ep = torch.export.export(
        model,
        tuple(encoded_input.values())
    )
    
    # Run decompositions
    ep = ep.run_decompositions()

    # Export to torch-mlir
    try:
        module = fx.export_and_import(
            ep, 
            **encoded_input,
            output_type="torch", 
            func_name="transformer"
        )
        return module, True
        
    except Exception as e:
        print(f"MLIR export failed: {e}")
        print("This might be due to unsupported operations in your model.")
        return None, False


# Usage example:
if __name__ == "__main__":
    #sentences = ["This is an example sentence", "Each sentence is converted"]

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    
    # Try the improved export
    module, success = export_to_mlir(model, tokenizer, max_length=64, batch_size=1)
    
    if success:
        with open("transformer_model_torch.mlir", "w") as f:
            f.write(str(module))
        print("✓ MLIR export successful!")
    else:
        print("Trying TorchScript fallback...")