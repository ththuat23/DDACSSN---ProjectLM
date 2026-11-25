from models.model_loader import load_model
from decoding.beam_search import beam_search_decode, calculate_beam_score

# Tải GPT-2 small
model, tokenizer = load_model()

# Prompt thử nghiệm
prompt = "Artificial intelligence is"


# 1. Kiểm thử Beam Search cơ bản
result = beam_search_decode(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    beam_width=5,
    length_penalty=0.6,
    min_new_tokens=0
)

print("Prompt:", prompt)
print("Generated text:", result)


# 2. Kiểm thử các giá trị Length Penalty
for alpha in [0.0, 0.6, 1.0]:

    print(f"\nLength Penalty (alpha) = {alpha}")

    result = beam_search_decode(
        model,
        tokenizer,
        prompt,
        max_new_tokens=70,
        beam_width=5,
        length_penalty=alpha,
        min_new_tokens=0
    )

    print("Generated text:", result)


# 3. Kiểm thử công thức Length Normalization
cumulative_score = -10.0
generated_length = 5

for alpha in [0.0, 0.6, 1.0]:

    score = calculate_beam_score(
        cumulative_score,
        generated_length,
        alpha
    )

    print(f"Alpha = {alpha}: Score = {score:.4f}")


# 4. Kiểm thử Minimum New Tokens
result = beam_search_decode(
    model,
    tokenizer,
    prompt,
    max_new_tokens=80,
    beam_width=5,
    length_penalty=0.6,
    min_new_tokens=60
)

generated_token_count = len(
    tokenizer.encode(result, add_special_tokens=False)
)

print("Minimum new tokens:", 60)
print("Generated token count:", generated_token_count)
print("Generated text:", result)
