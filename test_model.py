import torch
from models.model_loader import load_model

model, tokenizer = load_model()

prompt = "Artificial intelligence is"

device = next(model.parameters()).device

input_ids = tokenizer(
    prompt,
    return_tensors="pt"
)["input_ids"].to(device)

with torch.no_grad():
    logits = model(input_ids).logits

print("Logits shape:", logits.shape)
print("Device:", device)