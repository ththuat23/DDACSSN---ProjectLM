import time
import os
import psutil
import torch

from models.model_loader import load_model

def sync_device(device):
    """Đồng bộ GPU trước khi đo thời gian."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def measure_generation_performance(
    decode_function,
    model,
    tokenizer,
    prompt,
    **decode_kwargs
):
    """
    Đo hiệu suất của một chiến lược giải mã.

    Returns:
        generated_text: Văn bản được sinh.
        metrics: Các chỉ số hiệu suất.
    """

    device = next(model.parameters()).device
    process = psutil.Process(os.getpid())

    # Đo RAM trước khi sinh
    ram_before = process.memory_info().rss / (1024 ** 2)

    # Đo VRAM nếu sử dụng GPU
    if device.type == "cuda":
        sync_device(device)

        torch.cuda.reset_peak_memory_stats(device)

        vram_before = torch.cuda.memory_allocated(
            device
        ) / (1024 ** 2)
    else:
        vram_before = None

    # Bắt đầu đo thời gian
    sync_device(device)
    start_time = time.perf_counter()

    generated_text = decode_function(
        model,
        tokenizer,
        prompt,
        **decode_kwargs
    )

    sync_device(device)
    elapsed_time = time.perf_counter() - start_time

    # Đo RAM sau khi sinh
    ram_after = process.memory_info().rss / (1024 ** 2)

    # Đếm số token trong văn bản được sinh
    generated_tokens = len(
        tokenizer.encode(
            generated_text,
            add_special_tokens=False
        )
    )

    # Tính tốc độ sinh
    throughput = (
        generated_tokens / elapsed_time
        if elapsed_time > 0 else 0
    )

    # Đo VRAM
    if device.type == "cuda":
        vram_peak = torch.cuda.max_memory_allocated(
            device
        ) / (1024 ** 2)

        vram_extra_peak = max(
            0,
            vram_peak - vram_before
        )
    else:
        vram_peak = None
        vram_extra_peak = None

    metrics = {
        "generation_time_sec": elapsed_time,
        "generated_tokens": generated_tokens,
        "throughput_tokens_sec": throughput,
        "ram_before_mb": ram_before,
        "ram_after_mb": ram_after,
        "ram_delta_mb": ram_after - ram_before,
        "vram_peak_mb": vram_peak,
        "vram_extra_peak_mb": vram_extra_peak
    }

    return generated_text, metrics

if __name__ == "__main__":

    # Tải GPT-2 Small
    model, tokenizer = load_model()

    prompt = "Artificial intelligence is changing the world."
    
    # Import một chiến lược giải mã để kiểm tra
    from decoding.greedy import greedy_decode

    generated_text, metrics = measure_generation_performance(
        greedy_decode,
        model,
        tokenizer,
        prompt,
        max_new_tokens=30
    )

    print("\nGenerated text:")
    print(generated_text)

    print("\nPerformance metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
