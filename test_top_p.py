from models.model_loader import load_model
from decoding.top_p import top_p_decode

# Tải GPT-2
model, tokenizer = load_model()

# Prompt thử nghiệm
prompt = "Artificial intelligence is"

# Sinh văn bản bằng Top-P
result = top_p_decode(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    top_p=0.9
)

print("Prompt:", prompt)
print("Generated text:", result)
