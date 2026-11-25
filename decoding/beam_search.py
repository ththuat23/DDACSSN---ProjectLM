
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

def calculate_beam_score(cumulative_score, generated_length, length_penalty):
    """
    Chuẩn hóa điểm số theo độ dài chuỗi được sinh.
    """
    if generated_length <= 0:
        return cumulative_score

    return cumulative_score / (generated_length ** length_penalty)   

def beam_search_decode(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    beam_width=5,
    length_penalty=0.6,
    min_new_tokens=0
):
    # Kiểm tra tham số
    if beam_width <= 0:
        raise ValueError("beam_width phải lớn hơn 0")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens không được âm")

    if length_penalty < 0:
        raise ValueError("length_penalty không được âm")

    if min_new_tokens < 0 or min_new_tokens > max_new_tokens:
        raise ValueError("min_new_tokens phải nằm trong khoảng 0 đến max_new_tokens")

    model.eval()

    device = next(model.parameters()).device

    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt"
    ).to(device)
    # Lưu độ dài prompt để chỉ lấy phần văn bản mới sinh
    input_length = input_ids.shape[-1]

    # Mỗi beam gồm: (chuỗi token, tổng log probability, trạng thái kết thúc)
    beams = [(input_ids, 0.0, False)]

    eos_token_id = tokenizer.eos_token_id

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

                next_token_logits = logits[:, -1, :].clone()

                # Tính số token đã được sinh, không tính prompt
                generated_length = sequence.shape[-1] - input_length

                # Không cho phép kết thúc trước số token tối thiểu
                if (
                    eos_token_id is not None
                    and generated_length < min_new_tokens
                ):
                    next_token_logits[:, eos_token_id] = float("-inf")

                # Chọn các token có log probability cao nhất
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
                        eos_token_id is not None
                        and token_id.item() == eos_token_id
                    )

                    candidates.append(
                        (new_sequence, new_score, is_finished)
                    )

            # Giữ lại những beam có tổng điểm cao nhất
            beams = sorted(
                candidates,
                key=lambda x: calculate_beam_score(
                    x[1],
                    x[0].shape[-1] - input_length,
                    length_penalty
                ),
                reverse=True
            )[:beam_width]

            # Dừng nếu tất cả beam đã kết thúc
            if all(beam[2] for beam in beams):
                break

    # Chọn beam có điểm chuẩn hóa cao nhất
    best_beam = max(
        beams,
        key=lambda x: calculate_beam_score(
            x[1],
            x[0].shape[-1] - input_length,
            length_penalty
        )
    )

    generated_ids = best_beam[0][:, input_length:]

    return tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )