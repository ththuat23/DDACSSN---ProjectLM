import torch

def combined_from_logits(
    logits,
    temperature=0.8,
    top_k=None,
    top_p=None
):
    """
    Kết hợp Temperature với Top-k hoặc Top-p.
    Chỉ được chọn một trong hai: top_k hoặc top_p.
    """

    if temperature <= 0:
        raise ValueError("temperature phải lớn hơn 0")

    if (top_k is None) == (top_p is None):
        raise ValueError("Chỉ chọn top_k hoặc top_p")

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    if logits.dim() != 2:
        raise ValueError("Logits phải có 1 hoặc 2 chiều")

    # Bước 1: Temperature Scaling
    scaled_logits = logits / temperature

    # Bước 2: Lọc token bằng Top-k
    if top_k is not None:

        if top_k <= 0:
            raise ValueError("top_k phải lớn hơn 0")

        k = min(top_k, scaled_logits.size(-1))

        top_values, top_indices = torch.topk(
            scaled_logits,
            k=k,
            dim=-1
        )

        filtered_logits = torch.full_like(
            scaled_logits,
            float("-inf")
        )

        filtered_logits.scatter_(
            -1,
            top_indices,
            top_values
        )

    # Bước 2: Lọc token bằng Top-p
    else:

        if not 0 < top_p <= 1:
            raise ValueError("top_p phải nằm trong khoảng (0, 1]")

        probs = torch.softmax(scaled_logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(
            probs,
            descending=True,
            dim=-1
        )

        cumulative_probs = torch.cumsum(
            sorted_probs,
            dim=-1
        )

        # Loại token nằm ngoài tập xác suất tích lũy
        remove = cumulative_probs > top_p

        # Giữ lại token đầu tiên vượt ngưỡng
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False

        indices_to_remove = torch.zeros_like(
            remove,
            dtype=torch.bool
        )

        indices_to_remove.scatter_(
            -1,
            sorted_indices,
            remove
        )

        filtered_logits = scaled_logits.masked_fill(
            indices_to_remove,
            float("-inf")
        )

    # Bước 3: Chuẩn hóa xác suất và lấy mẫu
    final_probs = torch.softmax(
        filtered_logits,
        dim=-1
    )

    next_token = torch.multinomial(
        final_probs,
        num_samples=1
    )

    return next_token


def combined_decode(
    model,
    tokenizer,
    prompt,
    max_new_tokens=70,
    temperature=0.8,
    top_k=None,
    top_p=None
):
    """
    Sinh văn bản bằng Temperature kết hợp Top-k hoặc Top-p.
    """

    model.eval()

    device = next(model.parameters()).device

    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt"
    ).to(device)

    input_length = input_ids.shape[-1]

    with torch.no_grad():

        for _ in range(max_new_tokens):

            logits = model(input_ids).logits[:, -1, :]

            next_token = combined_from_logits(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=-1
            )

            if (
                tokenizer.eos_token_id is not None
                and next_token.item() == tokenizer.eos_token_id
            ):
                break

    generated_ids = input_ids[:, input_length:]

    return tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )