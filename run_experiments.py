import json
import os
import time
import torch

from pathlib import Path
from models.model_loader import load_model

from decoding.greedy import greedy_decode
from decoding.beam_search import beam_search_decode
from decoding.temperature import temperature_decode
from decoding.top_k import top_k_decode
from decoding.top_p import top_p_decode

# CẤU HÌNH THỬ NGHIỆM
BASE_DIR = Path(__file__).resolve().parent
PROMPT_FILE = BASE_DIR / "data" / "prompts.json"
OUTPUT_FILE = BASE_DIR / "results" / "smoke_test.json"



# GIỚI HẠN SỐ LƯỢNG PROMPT VÀ SỐ TOKEN MỚI
LIMIT = 5
MAX_NEW_TOKENS = 30
SEED = 42



# ĐỌC DỮ LIỆU
with open(PROMPT_FILE, "r", encoding="utf-8") as file:
    prompts = json.load(file)
prompts = prompts[:LIMIT]
print(f"So prompt se chay: {len(prompts)}")


# NẠP MODEL
model, tokenizer = load_model("gpt2")

device = next(model.parameters()).device
print(f"Model device: {device}")


# DANH SÁCH PHƯƠNG PHÁP
strategies = [
    {
        "name": "greedy",
        "function": greedy_decode,
        "parameters": {},
        "sampling": False
    },
    {
        "name": "beam_search",
        "function": beam_search_decode,
        "parameters": {"beam_width": 5},
        "sampling": False
    },
    {
        "name": "temperature",
        "function": temperature_decode,
        "parameters": {"temperature": 1.0},
        "sampling": True
    },
    {
        "name": "top_k",
        "function": top_k_decode,
        "parameters": {"top_k": 50},
        "sampling": True
    },
    {
        "name": "top_p",
        "function": top_p_decode,
        "parameters": {"top_p": 0.9},
        "sampling": True
    }
]


# CHẠY THỬ NGHIỆM
results = []

for prompt_item in prompts:

    prompt_id = prompt_item["id"]
    category = prompt_item["category"]
    prompt = prompt_item["prompt"]

    print("\n" + "=" * 60)
    print(f"Prompt ID: {prompt_id}")
    print(f"Category: {category}")
    print(f"Prompt: {prompt}")

    for strategy in strategies:

        name = strategy["name"]
        decode_function = strategy["function"]
        parameters = strategy["parameters"]

        # Cố định seed cho các phương pháp lấy mẫu
        if strategy["sampling"]:
            torch.manual_seed(SEED)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)

        start_time = time.perf_counter()

        generated_text = decode_function(
            model,
            tokenizer,
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            **parameters
        )

        elapsed_time = time.perf_counter() - start_time

        result = {
            "prompt_id": prompt_id,
            "category": category,
            "prompt": prompt,
            "strategy": name,
            "parameters": parameters,
            "generated_text": generated_text,
            "latency_seconds": round(elapsed_time, 4)
        }

        results.append(result)

        print(f"\n[{name}]")
        print(generated_text)
        print(f"Latency: {elapsed_time:.4f} seconds")


# LƯU KẾT QUẢ
os.makedirs(BASE_DIR / "results", exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(results, file, ensure_ascii=False, indent=4)
    
print("\n" + "=" * 60)
print("HOAN THANH THU NGHIEM")
print(f"So ket qua: {len(results)}")
print(f"Da luu tai: {OUTPUT_FILE}")
