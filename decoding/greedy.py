import torch


def greedy_from_logits(logits):
    # Chuyển vector [vocab_size] thành [1, vocab_size]
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    if logits.dim() != 2:
        raise ValueError("Logits phải có 1 hoặc 2 chiều")

    # Chọn token có logit lớn nhất
    next_token = torch.argmax(
        logits,
        dim=-1,
        keepdim=True
    )

    return next_token


def greedy_decode(model, tokenizer, prompt, max_new_tokens=50):
    model.eval()

    device = next(model.parameters()).device

    # Chuyển prompt thành ID token
    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt"
    ).to(device)

    # Lưu độ dài prompt ban đầu trước khi sinh token
    input_length = input_ids.shape[-1]

    # Sinh token liên tiếp
    with torch.no_grad():
        for _ in range(max_new_tokens):

            # Lấy logits trực tiếp từ mô hình
            logits = model(input_ids).logits

            # Lấy logits của token cuối cùng
            next_token_logits = logits[:, -1, :]

            next_token = greedy_from_logits(
                next_token_logits
            )

            # Thêm token mới vào chuỗi
            input_ids = torch.cat(
                [input_ids, next_token],
                dim=-1
            )

            # Dừng nếu gặp EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Chỉ lấy những token được sinh thêm
    generated_ids = input_ids[:, input_length:]

    return tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )
