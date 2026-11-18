import torch

def temperature_probs(logits, temperature):
    if temperature <= 0:
        raise ValueError("temperature phải lớn hơn 0")

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    if logits.dim() != 2:
        raise ValueError("Logits phải có 1 hoặc 2 chiều")

    # Điều chỉnh logits theo nhiệt độ
    scaled_logits = logits / temperature

    # Chuyển logits thành xác suất
    probs = torch.softmax(scaled_logits, dim=-1)

    return probs


def temperature_from_logits(logits, temperature):
    probs = temperature_probs(logits, temperature)

    # Lấy mẫu theo phân phối xác suất
    next_token = torch.multinomial(
        probs,
        num_samples=1
    )

    return next_token

def temperature_decode(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    temperature=1.0
):
    if temperature <= 0:
        raise ValueError("temperature phải lớn hơn 0")

    model.eval()

    device = next(model.parameters()).device

    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt"
    ).to(device)

    # Lưu độ dài prompt ban đầu trước khi sinh token
    input_length = input_ids.shape[-1]

    with torch.no_grad():
        for _ in range(max_new_tokens):

            # Lấy logits trực tiếp từ mô hình
            logits = model(input_ids).logits

            next_token_logits = logits[:, -1, :]

            # Điều chỉnh logits theo nhiệt độ
            scaled_logits = next_token_logits / temperature

            next_token = temperature_from_logits(
                next_token_logits,
                temperature
            )
            
            input_ids = torch.cat(
                [input_ids, next_token],
                dim=-1
            )

            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_ids = input_ids[:, input_length:]

    return tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )
