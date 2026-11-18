
import torch


def beam_search_from_logits(logits, beam_width):
    """
    Nhận logits và trả về beam_width token có log probability cao nhất.
    """

    if beam_width <= 0:
        raise ValueError("beam_width phải lớn hơn 0")

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    if logits.dim() != 2:
        raise ValueError("Logits phải có 1 hoặc 2 chiều")

    # Chuyển logits thành log probability
    log_probs = torch.log_softmax(
        logits,
        dim=-1
    )

    # Chọn các token có log probability cao nhất
    top_log_probs, top_tokens = torch.topk(
        log_probs,
        k=min(beam_width, log_probs.size(-1)),
        dim=-1
    )

    return top_log_probs, top_tokens


def beam_search_decode(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    beam_width=5
):
    if beam_width <= 0:
        raise ValueError("beam_width phải lớn hơn 0")

    model.eval()

    device = next(model.parameters()).device

    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt"
    ).to(device)

    # Mỗi beam gồm: (chuỗi token, tổng log probability, trạng thái kết thúc)
    beams = [(input_ids, 0.0, False)]

    with torch.no_grad():
        for _ in range(max_new_tokens):

            candidates = []

            for sequence, score, finished in beams:

                # Giữ nguyên beam đã kết thúc
                if finished:
                    candidates.append(
                        (sequence, score, True)
                    )
                    continue

                # Lấy logits trực tiếp từ mô hình
                logits = model(sequence).logits

                next_token_logits = logits[:, -1, :]

                top_log_probs, top_tokens = beam_search_from_logits(
                    next_token_logits,
                    beam_width
                )

                for log_prob, token_id in zip(
                    top_log_probs[0],
                    top_tokens[0]
                ):
                    next_token = token_id.view(1, 1)

                    new_sequence = torch.cat(
                        [sequence, next_token],
                        dim=-1
                    )

                    new_score = score + log_prob.item()

                    is_finished = (
                        tokenizer.eos_token_id is not None
                        and token_id.item() == tokenizer.eos_token_id
                    )

                    candidates.append(
                        (new_sequence, new_score, is_finished)
                    )

            # Giữ lại những beam có tổng điểm cao nhất
            beams = sorted(
                candidates,
                key=lambda x: x[1],
                reverse=True
            )[:beam_width]

            # Dừng nếu tất cả beam đã kết thúc
            if all(beam[2] for beam in beams):
                break

    # Chọn beam có điểm cao nhất
    best_beam = max(beams, key=lambda x: x[1])

    input_length = input_ids.shape[-1]

    generated_ids = best_beam[0][:, input_length:]

    return tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )