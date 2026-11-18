from models.model_loader import load_model
from decoding.greedy import greedy_decode

# Tải GPT-2
model, tokenizer = load_model()

# Prompt thử nghiệm
prompt = "Artificial intelligence is"

# Sinh văn bản bằng Greedy
result = greedy_decode(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50
)

print("Prompt:", prompt)
print("Generated text:", result)
