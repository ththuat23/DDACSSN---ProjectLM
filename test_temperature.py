from models.model_loader import load_model
from decoding.temperature import temperature_decode

# Tải GPT-2
model, tokenizer = load_model()

# Prompt thử nghiệm
prompt = "Artificial intelligence is"

# Sinh văn bản bằng Temperature
result = temperature_decode(
    model,
    tokenizer,
    prompt,
    temperature=0.7
)

print("Prompt:", prompt)
print("Generated text:", result)
