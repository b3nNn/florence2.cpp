from transformers import BartTokenizerFast, CLIPImageProcessor
from PIL import Image
import numpy as np

class Florence2Processor:
    def __init__(self, tokenizer=None, image_processor=None):
        # Use BartTokenizerFast explicitly instead of AutoTokenizer
        self.tokenizer = tokenizer if tokenizer is not None else BartTokenizerFast.from_pretrained("microsoft/florence-2-large-ft")
        self.image_processor = image_processor if image_processor is not None else CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        self.task_prompts = {
            "<OCR>": "<OCR>",
            "<CAPTION>": "<CAPTION>",
            "<DETAILED_CAPTION>": "<DETAILED_CAPTION>",
            "<MORE_DETAILED_CAPTION>": "<MORE_DETAILED_CAPTION>",
            "<OD>": "<OD>",
            "<REGION_CAPTION>": "<REGION_CAPTION>",
            "<CAPTION_TO_PHRASE_GROUNDING>": "<CAPTION_TO_PHRASE_GROUNDING>",
            "<REFERRAL_EXPRESSION_GROUNDING>": "<REFERRAL_EXPRESSION_GROUNDING>",
            "<DENSE_REGION_CAPTION>": "<DENSE_REGION_CAPTION>",
            "<REGION_PROPOSAL>": "<REGION_PROPOSAL>",
        }

    def __call__(self, text=None, images=None, task_prompt=None, return_tensors="np"):
        if images is None:
            raise ValueError("Florence2Processor expects at least one image as input.")
        if task_prompt is None:
            raise ValueError("Florence2Processor expects a task_prompt.")
        if task_prompt not in self.task_prompts:
            raise ValueError(f"Task prompt {task_prompt} not recognized.")

        # Handle single image or list of images
        if isinstance(images, (Image.Image, np.ndarray)):
            images = [images]

        # Process images
        image_inputs = self.image_processor(images=images, return_tensors=return_tensors)
        pixel_values = image_inputs["pixel_values"]

        # Process text with task prompt
        if text is None:
            text = ""
        full_text = self.task_prompts[task_prompt] + text
        text_inputs = self.tokenizer(full_text, return_tensors=return_tensors, padding=True, truncation=True)

        return {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "pixel_values": pixel_values,
        }

def test_florence2_processor(image_path="puma.png", task_prompt="<CAPTION>"):
    processor = Florence2Processor()
    image = Image.open(image_path).convert("RGB")
    print(f"Original image size: {image.size}")

    inputs = processor(text="", images=image, task_prompt=task_prompt)
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]

    print(f"Input IDs: {input_ids.flatten().tolist()}")
    print(f"Pixel values shape: {pixel_values.shape}")
    print(f"First few values (channel 0, top-left corner): {pixel_values[0, 0, :5, :5]}")
    print(f"Mean across channels: {pixel_values.mean(axis=(0, 2, 3))}")
    print(f"Std across channels: {pixel_values.std(axis=(0, 2, 3))}")

    # Simulate model output
    sample_caption = "A beautiful sunset over the ocean."
    output_ids = processor.tokenizer.encode(sample_caption, add_special_tokens=True)
    print(f"Simulated output IDs for '{sample_caption}': {output_ids}")
    output_text = processor.tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"Decoded output: {output_text}")

if __name__ == "__main__":
    try:
        test_florence2_processor("puma.png", "<CAPTION>")
    except Exception as e:
        print(f"Error: {e}")