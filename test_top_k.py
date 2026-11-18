from models.model_loader import load_model
from decoding.top_k import top_k_decode

# Tải GPT-2
model, tokenizer = load_model()

# Prompt thử nghiệm
prompt = "Artificial intelligence is"

# Sinh văn bản bằng Top-K
result = top_k_decode(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    top_k=5
)

print("Prompt:", prompt)
print("Generated text:", result)
