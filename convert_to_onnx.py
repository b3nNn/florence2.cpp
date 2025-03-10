import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BartTokenizerFast
from PIL import Image
import numpy as np

# Load model and processor
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)

# Move model to evaluation mode
model.eval()

# Custom wrapper with nested config access
class Florence2ONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Try language model config first, fallback to default
        try:
            self.max_position_embeddings = model.language_model.config.max_position_embeddings
        except AttributeError:
            print("Warning: max_position_embeddings not found, using default 512")
            self.max_position_embeddings = 512  # Safe default based on BART
        # Access positional embedding size directly
        self.pos_embed_size = model.language_model.model.encoder.embed_positions.num_embeddings
        print(f"Max position embeddings (config): {self.max_position_embeddings}")
        print(f"Positional embedding size: {self.pos_embed_size}")

    def forward(self, pixel_values, input_ids, attention_mask, decoder_input_ids):
        input_ids = input_ids.long()
        attention_mask = attention_mask.long()
        decoder_input_ids = decoder_input_ids.long()

        # Cap sequence length to fit positional embeddings
        safe_seq_len = min(input_ids.size(1), min(self.max_position_embeddings, self.pos_embed_size) - 2)
        input_ids = input_ids[:, :safe_seq_len]
        attention_mask = attention_mask[:, :safe_seq_len]
        decoder_input_ids = decoder_input_ids[:, :safe_seq_len]

        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )
        return outputs["logits"]

wrapped_model = Florence2ONNXWrapper(model)

# Dummy text and image
dummy_text = "<CAPTION>"
dummy_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
dummy_image = Image.fromarray(dummy_image)

# Pre-tokenize with Florence-2's tokenizer
max_len = 256  # Conservative length
tokenizer = processor.tokenizer
input_ids = tokenizer.encode(dummy_text, add_special_tokens=True, max_length=max_len, truncation=True, padding="max_length", return_tensors="pt")
attention_mask = torch.ones_like(input_ids, dtype=torch.int64)

# Dummy decoder_input_ids
bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
decoder_input_ids = torch.full((1, max_len), bos_token_id, dtype=torch.int64)

# Preprocess image
image_inputs = processor(images=dummy_image, return_tensors="pt")
pixel_values = image_inputs["pixel_values"]  # Shape: [1, 3, 224, 224]

# Ensure correct dtypes
input_ids = input_ids.to(torch.int64)
attention_mask = attention_mask.to(torch.int64)
decoder_input_ids = decoder_input_ids.to(torch.int64)

# Export to ONNX
torch.onnx.export(
    wrapped_model,
    (pixel_values, input_ids, attention_mask, decoder_input_ids),
    "florence2.onnx",
    input_names=["pixel_values", "input_ids", "attention_mask", "decoder_input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "decoder_input_ids": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size"}
    },
    opset_version=14,
    verbose=True
)

print("Model exported to florence2.onnx")