import torch

def top_p_filter_logits(logits, p):
    if not 0 < p <= 1:
        raise ValueError("top_p phải nằm trong khoảng (0, 1]")

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    if logits.dim() != 2:
        raise ValueError("Logits phải có 1 hoặc 2 chiều")

    if p == 1:
        return logits

    # Chuyển logits thành xác suất
    probs = torch.softmax(logits, dim=-1)

    # Sắp xếp xác suất giảm dần
    sorted_probs, sorted_indices = torch.sort(
        probs,
        descending=True,
        dim=-1
    )

    # Tính xác suất tích lũy
    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=-1
    )

    # Xác định token cần loại bỏ
    sorted_indices_to_remove = cumulative_probs > p

    # Giữ lại token đầu tiên vượt ngưỡng p
    sorted_indices_to_remove[..., 1:] = (
        sorted_indices_to_remove[..., :-1].clone()
    )

    sorted_indices_to_remove[..., 0] = False

    # Đưa mặt nạ về thứ tự token ban đầu
    indices_to_remove = torch.zeros_like(
        probs,
        dtype=torch.bool
    )

    indices_to_remove.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove
    )

    # Loại bỏ các token không thuộc top-p
    filtered_logits = logits.masked_fill(
        indices_to_remove,
        float("-inf")
    )

    return filtered_logits


def top_p_probs(logits, p):
    filtered_logits = top_p_filter_logits(logits, p)

    probs = torch.softmax(
        filtered_logits,
        dim=-1
    )

    return probs


def top_p_from_logits(logits, p):
    probs = top_p_probs(logits, p)

    next_token = torch.multinomial(
        probs,
        num_samples=1
    )

    return next_token

def top_p_decode(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    top_p=0.9
):
    if not 0 < top_p <= 1:
        raise ValueError("top_p phải nằm trong khoảng (0, 1]")

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

            next_token = top_p_from_logits(
                next_token_logits,
                top_p
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
