import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

# Load model and processor
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)

class Florence2ONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, input_ids, decoder_input_ids):
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids
        )
        return outputs["logits"]

wrapped_model = Florence2ONNXWrapper(model)

# Real image (match runtime)
dummy_image = Image.open("puma.png").resize((768, 768))  # Match CLIPImageProcessor
image_inputs = processor(images=dummy_image, return_tensors="pt")
pixel_values = image_inputs["pixel_values"]  # [1, 3, 768, 768]

# Prompt
dummy_text = "<CAPTION>"
input_ids = processor.tokenizer.encode(dummy_text, add_special_tokens=True, max_length=256, truncation=True, padding="max_length", return_tensors="pt")
decoder_input_ids = torch.full((1, 256), processor.tokenizer.bos_token_id or 0, dtype=torch.int64)

# Export with matching inputs
torch.onnx.export(
    wrapped_model,
    (pixel_values, input_ids, decoder_input_ids),
    "florence2.onnx",
    input_names=["pixel_values", "input_ids", "decoder_input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "decoder_input_ids": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size", 1: "seq_len"}
    },
    opset_version=14,
    verbose=True
)

print("Model re-exported to florence2.onnx")
