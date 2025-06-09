import torch
from cnn_model import ViewportPredictor

model = ViewportPredictor()
tokenizer = model.tokenizer

prompt = "The past 5 viewports were:\n(1.0,2.0,3.0)\n(2.0,3.0,4.0)\n(3.0,4.0,5.0)\n(4.0,5.0,6.0)\n(5.0,6.0,7.0)\nWhat are the next 5 viewports?"

inputs = tokenizer(prompt, return_tensors="pt")
output = model.model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
