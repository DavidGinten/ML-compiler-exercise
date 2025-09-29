from transformers import T5Tokenizer, T5EncoderModel
import torch
from torch_mlir import fx

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5EncoderModel.from_pretrained("google/flan-t5-small")

input_text = "translate English to German: How old are you?"
encoded_input = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

wrapped_model = Wrapper(model)

# Export with torch.export
ep = torch.export.export(
    wrapped_model,
    (
        encoded_input["input_ids"],
        encoded_input["attention_mask"],
    )
)

ep = ep.run_decompositions()
m = fx.export_and_import(
    ep,
    output_type=fx.OutputType.LINALG_ON_TENSORS,
    func_name="transformer_model"
)

mlir_str = str(m)
with open("google_linalg.mlir", "w") as f:
    f.write(mlir_str)
