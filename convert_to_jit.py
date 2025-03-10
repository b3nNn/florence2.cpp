import torch
from transformers import AutoModelForCausalLM

output_path = "./florence2_jit.pt"

# Load the model
model = AutoModelForCausalLM.from_pretrained("microsoft/florence-2-large-ft", trust_remote_code=True)
model.eval()

# Custom forward to return logits only
def custom_forward(self, pixel_values, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
    input_ids = input_ids.to(torch.int64)
    attention_mask = attention_mask.to(torch.int64)
    decoder_input_ids = decoder_input_ids.to(torch.int64)
    decoder_attention_mask = decoder_attention_mask.to(torch.int64)
    output = self.__class__.forward(
        self,
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask
    )
    return output.logits  # Return only the logits tensor

# Patch the forward method
model.forward = custom_forward.__get__(model, type(model))

# Example inputs
pixel_values = torch.randn(1, 3, 224, 224, dtype=torch.float32)
input_ids = torch.randint(0, model.config.vocab_size, (1, 10), dtype=torch.int64)
attention_mask = torch.ones(1, 10, dtype=torch.int64)
decoder_input_ids = torch.randint(0, model.config.vocab_size, (1, 5), dtype=torch.int64)
decoder_attention_mask = torch.ones(1, 5, dtype=torch.int64)

# Debug: Print input types and shapes
print("pixel_values:", pixel_values.dtype, pixel_values.shape)
print("input_ids:", input_ids.dtype, input_ids.shape)
print("attention_mask:", attention_mask.dtype, attention_mask.shape)
print("decoder_input_ids:", decoder_input_ids.dtype, decoder_input_ids.shape)
print("decoder_attention_mask:", decoder_attention_mask.dtype, decoder_attention_mask.shape)

# Trace the model
try:
    traced_model = torch.jit.trace(
        model,
        (pixel_values, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask),
        strict=False
    )
    traced_model.save(output_path)
    print(f"JIT model saved to {output_path}")
except Exception as e:
    print(f"Tracing failed: {str(e)}")

# Manual forward pass
try:
    with torch.no_grad():
        logits = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask
        )
    print("Manual forward pass succeeded, logits shape:", logits.shape)
except Exception as e:
    print(f"Manual forward failed: {str(e)}")