from transformers import CLIPImageProcessor
from PIL import Image
import numpy as np

def test_clip_processor(image_path="puma.png"):
    # Initialize the CLIPImageProcessor with default settings
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load the image
    image = Image.open(image_path).convert("RGB")
    print(f"Original image size: {image.size}")  # (width, height)

    # Process the image
    processed = processor(images=image, return_tensors="np")
    pixel_values = processed["pixel_values"]  # Shape: [batch, channels, height, width]

    # Print shape and some values for comparison
    print(f"Processed image shape: {pixel_values.shape}")
    print(f"First few values (channel 0, top-left corner): {pixel_values[0, 0, :5, :5]}")
    print(f"Mean across channels: {pixel_values.mean(axis=(0, 2, 3))}")
    print(f"Std across channels: {pixel_values.std(axis=(0, 2, 3))}")

if __name__ == "__main__":
    try:
        test_clip_processor("puma.png")
    except Exception as e:
        print(f"Error: {e}")