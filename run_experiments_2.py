import random
import json
import torch

from pathlib import Path

from models.model_loader import load_model

from decoding.greedy import greedy_decode
from decoding.beam_search import beam_search_decode
from decoding.temperature import temperature_decode
from decoding.top_k import top_k_decode
from decoding.top_p import top_p_decode
from decoding.combined import combined_decode

from evaluation.performance import measure_generation_performance

# CẤU HÌNH THỬ NGHIỆM
BASE_DIR = Path(__file__).resolve().parent

PROMPT_FILE = BASE_DIR / "data" / "prompts.json"
RESULT_FILE = BASE_DIR / "results" / "raw_results.json"

PROMPT_LIMIT = 20
MAX_NEW_TOKENS = 70

SEEDS = [42, 123, 2026]

# ĐỌC PROMPT
def load_prompts(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    records = []

    def add_item(item, category=None):

        if isinstance(item, str):
            prompt = item
            prompt_id = len(records) + 1

        elif isinstance(item, dict):

            prompt = (
                item.get("prompt")
                or item.get("text")
                or item.get("input")
            )

            if not prompt:
                return

            prompt_id = item.get(
                "id",
                len(records) + 1
            )

            category = item.get("category", category)

        else:
            return

        records.append({
            "id": prompt_id,
            "category": category or "uncategorized",
            "prompt": prompt
        })

    if isinstance(data, list):

        for item in data:
            add_item(item)

    elif isinstance(data, dict):

        if isinstance(data.get("prompts"), list):

            for item in data["prompts"]:
                add_item(item)

        else:

            for category, items in data.items():

                if isinstance(items, list):

                    for item in items:
                        add_item(item, category)

    if not records:
        raise ValueError(
            "Không tìm thấy prompt hợp lệ trong prompts.json"
        )

    return records



def select_prompts(records, limit=20, seed=42):

    import random

    rng = random.Random(seed)

    # Gom prompt theo category
    grouped = {}

    for item in records:
        category = item["category"]
        grouped.setdefault(category, []).append(item)

    if limit > len(records):
        raise ValueError("Số lượng prompt yêu cầu vượt quá dữ liệu")

    categories = list(grouped.keys())

    # Tính số prompt cần chọn ở mỗi category
    base = limit // len(categories)
    remainder = limit % len(categories)

    # Xáo trộn thứ tự category
    rng.shuffle(categories)

    selected = []

    for index, category in enumerate(categories):

        count = base + (1 if index < remainder else 0)

        if count > len(grouped[category]):
            raise ValueError(
                f"Category '{category}' không đủ prompt"
            )

        # Chọn ngẫu nhiên, không trùng lặp
        selected.extend(
            rng.sample(grouped[category], count)
        )

    # Xáo trộn thứ tự prompt
    rng.shuffle(selected)

    return selected


# THIẾT LẬP SEED
def set_seed(seed):

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# LƯU KẾT QUẢ
def save_results(results):

    RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(RESULT_FILE, "w", encoding="utf-8") as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=4
        )

# CHẠY THỬ NGHIỆM
def main():

    print("Loading GPT-2 Small...")

    model, tokenizer = load_model("gpt2")

    prompts = select_prompts(
        load_prompts(PROMPT_FILE),
        PROMPT_LIMIT
    )

    print("Number of prompts:", len(prompts))

    # Tên phương pháp, hàm giải mã, tham số, có lấy mẫu hay không
    strategies = [

        (
            "Greedy",
            greedy_decode,
            {},
            False
        ),

        (
            "Beam Search",
            beam_search_decode,
            {
                "beam_width": 5,
                "length_penalty": 0.6,
                "min_new_tokens": 0
            },
            False
        ),

        (
            "Temperature",
            temperature_decode,
            {
                "temperature": 0.8
            },
            True
        ),

        (
            "Top-k",
            top_k_decode,
            {
                "top_k": 50
            },
            True
        ),

        (
            "Top-p",
            top_p_decode,
            {
                "top_p": 0.9
            },
            True
        ),

        (
            "Temperature + Top-k",
            combined_decode,
            {
                "temperature": 0.8,
                "top_k": 50
            },
            True
        ),

        (
            "Temperature + Top-p",
            combined_decode,
            {
                "temperature": 0.8,
                "top_p": 0.9
            },
            True
        )
    ]

    results = []

    total = sum(
        len(SEEDS) if stochastic else 1
        for _, _, _, stochastic in strategies
    ) * len(prompts)

    current = 0

    for prompt_item in prompts:

        prompt_id = prompt_item["id"]
        category = prompt_item["category"]
        prompt = prompt_item["prompt"]

        print("\n" + "=" * 60)
        print(f"Prompt ID: {prompt_id}")
        print(f"Category: {category}")
        print(f"Prompt: {prompt}")
        print("=" * 60)

        for name, decode_function, kwargs, stochastic in strategies:

            seeds = SEEDS if stochastic else [None]

            for seed in seeds:

                current += 1

                if seed is not None:
                    set_seed(seed)

                print(
                    f"\n[{current}/{total}] "
                    f"{name} | Seed: {seed}"
                )

                generated_text, performance = measure_generation_performance(
                    decode_function,
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    **kwargs
                )

                record = {
                    "prompt_id": prompt_item["id"],
                    "category": prompt_item["category"],
                    "prompt": prompt,
                    "method": name,
                    "seed": seed,
                    "generated_text": generated_text,
                    "performance": performance,
                    "parameters": kwargs
                }

                results.append(record)

                # Lưu sau mỗi lần sinh để tránh mất kết quả
                save_results(results)

                print("Generated text:")
                print(generated_text)
                print(
                    "Latency:",
                    f"{performance['generation_time_sec']:.4f} seconds"
                )

    print("\n" + "=" * 55)
    print("GENERATION COMPLETED")
    print("Results saved to:", RESULT_FILE)
    print("=" * 55)


if __name__ == "__main__":
    main()