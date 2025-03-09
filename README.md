# bart.cpp

The goal of this project is to produce a C++ version of [BART](https://huggingface.co/facebook/bart-large)'s tokenizer. The original version was released for Python using `torch` and [the implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/tokenization_bart.py) is available under `Apache 2.0` licence.

_To be continued..._

## GGUF

First, save the model.
```python
from transformers import BartTokenizerFast
model = BartTokenizerFast.from_pretrained("microsoft/Florence-2-large-ft")
model.save_pretrained("florence2")
```

Then, convert to gguf.
```shell
convert_to_gguf.py florence2 --outfile florence2-large-ft.gguf --outtype f16
```
