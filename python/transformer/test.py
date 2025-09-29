# --- encoder export ---
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from torch_mlir import fx
from torch_mlir.fx import OutputType

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
full = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")


# --- encoder wrapper ---
class EncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.get_encoder()  # T5 encoder module
    def forward(self, input_ids, attention_mask):
        # returns encoder_hidden_states (last_hidden_state)
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state

enc_wrapped = EncoderWrapper(full)

# provide a fixed-length encoder example (pad/truncate to N)
encoder_input = tokenizer("translate English to German: How old are you?",
                          return_tensors="pt", padding="max_length", max_length=11, truncation=True)
ep_enc = torch.export.export(
    enc_wrapped,
    (encoder_input["input_ids"], encoder_input["attention_mask"]),
    # keep encoder shapes static here to avoid symbolic leaks
)
ep_enc = ep_enc.run_decompositions()
m_enc = fx.export_and_import(ep_enc, output_type=OutputType.LINALG_ON_TENSORS, func_name="t5_encoder")
mlir_str = str(m_enc)
open("t5_encoder.mlir","w").write(mlir_str)


# --- decoder-step wrapper ---
class DecoderStepWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # use internal decoder module; use embed + decoder + lm_head to get logits
        self.decoder = model.get_decoder()
        self.embed_tokens = model.shared  # token embeddings (tied)
        self.lm_head = model.lm_head       # final vocab projection
    def forward(self, last_token_ids, encoder_hidden_states, encoder_attention_mask):
        # last_token_ids: [1,1] static
        # encoder_hidden_states: [1, src_len, hidden]
        # embed last token
        #x = self.embed_tokens(last_token_ids)        # [1,1,hidden]
        # call decoder: note T5 decoder expects encoder_hidden_states keyword
        # attention_mask handling may be required depending on version; pass encoder_attention_mask if needed
        dec_out = self.decoder(input_ids=last_token_ids,
                               encoder_hidden_states=encoder_hidden_states,
                               encoder_attention_mask=encoder_attention_mask)
        # dec_out[0] or dec_out.last_hidden_state depending on HF version
        last_hidden = dec_out.last_hidden_state  # [1,1,hidden]
        logits = self.lm_head(last_hidden)       # [1,1,vocab_size]
        return logits

dec_wrapped = DecoderStepWrapper(full)

# example shapes: last token [1,1], encoder_hidden_states [1,11,hidden_dim]
last_token = torch.tensor([[tokenizer.pad_token_id, tokenizer.pad_token_id]])
encoder_hidden_example = torch.randn(1, 11, full.config.d_model)
encoder_mask_example = encoder_input["attention_mask"]
dec_len = torch.export.Dim("dec_len", max=16)

ep_dec = torch.export.export(
    dec_wrapped,
    (last_token, encoder_hidden_example, encoder_mask_example),
    dynamic_shapes=(
        {0:1, 1:dec_len},                # last_token fixed shape [1,1]
        {0:1, 1: 11},              # encoder_hidden_states: keep src_len static (or set dynamic if you can)
        {0:1, 1:11},               # encoder_attention_mask static
    )
)
ep_dec = ep_dec.run_decompositions()
m_dec = fx.export_and_import(ep_dec, output_type=OutputType.LINALG_ON_TENSORS, func_name="t5_decoder_step")
open("t5_decoder_step.mlir","w").write(str(m_dec))
