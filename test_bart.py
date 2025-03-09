from transformers import BartTokenizerFast
tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
text = "Hello, world! This is my final test to ensure that the tokenizer is working propertly. Thanks to grok for the help."
encoded = tokenizer.encode(text, add_special_tokens=True)
print("Token IDs:", encoded)
decoded = tokenizer.decode(encoded, skip_special_tokens=False)
print("Decoded text:", decoded)