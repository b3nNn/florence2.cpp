import torch
model = torch.load("G:\Downloads\pytorch_model.bin", map_location="cpu")
print(list(model.keys()))