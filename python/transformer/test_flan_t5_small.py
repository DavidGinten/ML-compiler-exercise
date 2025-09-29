from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

input_text = "translate English to German: A step by step recipe to make bolognese pasta:"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
#print(outputs[0])
print(tokenizer.decode([2729, 15, 7, 316, 2751, 2747, 2743, 2739, 2751, 2747]))


"""
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

tok = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

input_ids = tok("translate English to German: How old are you?", return_tensors="pt").input_ids
decoder_input_ids = torch.tensor([[tok.pad_token_id]])

with torch.no_grad():
    out = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    logits = out.logits  # shape [1,1,32128]
    token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
    print("Next token ID:", token_id, "=", tok.decode([token_id]))
"""