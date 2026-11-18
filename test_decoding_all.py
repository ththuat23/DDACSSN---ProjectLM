import time
import torch

from models.model_loader import load_model

from decoding.greedy import greedy_decode
from decoding.beam_search import beam_search_decode
from decoding.temperature import temperature_decode
from decoding.top_k import top_k_decode
from decoding.top_p import top_p_decode


# TẢI MODEL
model, tokenizer = load_model()

prompt = "Artificial intelligence is"
max_tokens = 50

# HÀM ĐO THỜI GIAN
def test_strategy(name, decode_function, **kwargs):
    print(f"\n{name} -")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    generated_text = decode_function(
        model, tokenizer, prompt,
        max_new_tokens=max_tokens,
        **kwargs
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latency = time.perf_counter() - start_time

    print("Generated text:")
    print(generated_text)
    print(f"Latency: {latency:.4f} seconds")

    return latency


# CHẠY 5 PHƯƠNG PHÁP
results = {}

results["Greedy"] = test_strategy(
    "GREEDY",
    greedy_decode
)

results["Beam Search"] = test_strategy(
    "BEAM SEARCH",
    beam_search_decode,
    beam_width=5
)

results["Temperature"] = test_strategy(
    "TEMPERATURE",
    temperature_decode,
    temperature=0.8
)

results["Top-k"] = test_strategy(
    "TOP-K",
    top_k_decode,
    top_k=50
)

results["Top-p"] = test_strategy(
    "TOP-P",
    top_p_decode,
    top_p=0.9
)

# # TỔNG HỢP THỜI GIAN
# print("\n" + "=" * 45)
# print("TỔNG HỢP THỜI GIAN SINH VĂN BẢN")
# print("=" * 45)

# for name, elapsed in results.items():
#     print(f"{name:<15}: {elapsed:.4f} seconds")

# print("=" * 45)