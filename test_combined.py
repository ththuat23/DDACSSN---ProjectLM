from models.model_loader import load_model
from decoding.combined import combined_decode

# TẢI MODEL
model, tokenizer = load_model()

prompt = "Artificial intelligence is"
max_tokens = 50

# Temperature + Top-k
text_topk_temp = combined_decode(
    model,
    tokenizer,
    prompt,
    max_new_tokens=70,
    temperature=0.8,
    top_k=50
)

# Temperature + Top-p
text_topp_temp = combined_decode(
    model,
    tokenizer,
    prompt,
    max_new_tokens=70,
    temperature=0.8,
    top_p=0.9
)

print("Prompt:", prompt)
print("\nGenerated text (Top-k + Temperature):", text_topk_temp)
print("\nGenerated text (Top-p + Temperature):", text_topp_temp)
