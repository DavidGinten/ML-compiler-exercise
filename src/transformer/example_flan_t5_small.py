from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

input_text = "translate English to German: How are you?"
input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).input_ids

print(input_ids)

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
print(outputs[0])

# Test decoding a specific token ID sequence
print(tokenizer.decode([0, 2739, 229, 3, 15, 7]))