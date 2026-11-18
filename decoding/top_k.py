import torch

def top_k_filter_logits(logits, k):
    if k <= 0:
        raise ValueError("top_k phải lớn hơn 0")

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    if logits.dim() != 2:
        raise ValueError("Logits phải có 1 hoặc 2 chiều")

    k = min(k, logits.size(-1))

    # Lấy k logits lớn nhất
    top_logits, top_indices = torch.topk(
        logits,
        k=k,
        dim=-1
    )

    # Loại bỏ các token không thuộc top-k
    filtered_logits = torch.full_like(
        logits,
        float("-inf")
    )

    filtered_logits.scatter_(
        dim=-1,
        index=top_indices,
        src=top_logits
    )

    return filtered_logits


def top_k_probs(logits, k):
    filtered_logits = top_k_filter_logits(logits, k)

    probs = torch.softmax(
        filtered_logits,
        dim=-1
    )

    return probs


def top_k_from_logits(logits, k):
    probs = top_k_probs(logits, k)

    next_token = torch.multinomial(
        probs,
        num_samples=1
    )

    return next_token

def top_k_decode(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    top_k=50
):
    if top_k <= 0:
        raise ValueError("top_k phải lớn hơn 0")

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

            next_token = top_k_from_logits(
                next_token_logits,
                top_k
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
