import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

# Load model and processor
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
model.eval()

# Prepare inputs manually
image = Image.open("puma.png").convert("RGB")
inputs = processor(images=image, text="<CAPTION>", return_tensors="pt")
pixel_values = inputs["pixel_values"]  # [1, 3, 768, 768], float32
input_ids = inputs["input_ids"].to(torch.long)  # [1, seq_len], int64
decoder_input_ids = torch.ones((1, 1), dtype=torch.long) * processor.tokenizer.bos_token_id  # [1, 1], int64

# Debug tensor types
print("pixel_values dtype:", pixel_values.dtype)
print("input_ids dtype:", input_ids.dtype)
print("decoder_input_ids dtype:", decoder_input_ids.dtype)

# Wrapper for tracing
class Florence2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, input_ids, decoder_input_ids):
        print("Wrapper - pixel_values dtype:", pixel_values.dtype)
        print("Wrapper - input_ids dtype:", input_ids.dtype)
        print("Wrapper - decoder_input_ids dtype:", decoder_input_ids.dtype)
        return self.model(pixel_values=pixel_values, input_ids=input_ids, decoder_input_ids=decoder_input_ids)["logits"]

wrapped_model = Florence2Wrapper(model)

# Dummy forward pass
with torch.no_grad():
    outputs = wrapped_model(pixel_values, input_ids, decoder_input_ids)

# Export to ONNX
torch.onnx.export(
    wrapped_model,
    (pixel_values, input_ids, decoder_input_ids),
    "florence2.onnx",
    opset_version=14,
    input_names=["pixel_values", "input_ids", "decoder_input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "decoder_input_ids": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size", 1: "seq_len"}
    }
)

print("Exported florence2.onnx successfully")