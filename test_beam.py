from models.model_loader import load_model
from decoding.beam_search import beam_search_decode

# Tải GPT-2
model, tokenizer = load_model()

# Prompt thử nghiệm
prompt = "Artificial intelligence is"

# Sinh văn bản bằng Beam Search
result = beam_search_decode(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    beam_width=5
)

print("Prompt:", prompt)
print("Generated text:", result)
