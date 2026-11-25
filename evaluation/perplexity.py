
import torch
import math

from models.model_loader import load_medium_model


def calculate_perplexity(
    model,
    tokenizer,
    prompt,
    generated_text
):
    """
    Tính Perplexity trên phần văn bản được sinh.

    model: GPT-2 Medium
    tokenizer: Tokenizer của GPT-2 Medium
    prompt: Prompt ban đầu
    generated_text: Văn bản mới sinh
    """

    if not generated_text.strip():
        raise ValueError("Văn bản sinh ra đang rỗng")

    if not tokenizer.is_fast:
        raise ValueError("Tokenizer phải hỗ trợ offset_mapping")

    device = next(model.parameters()).device

    # Ghép prompt và phần văn bản được sinh
    full_text = prompt + generated_text

    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False
    )

    input_ids = inputs["input_ids"].to(device)
    offsets = inputs["offset_mapping"][0]

    # Tạo labels để tính loss
    labels = input_ids.clone()

    # Bỏ qua token thuộc prompt và token giao với ranh giới prompt
    for i, (start, end) in enumerate(offsets.tolist()):
        if start < len(prompt):
            labels[0, i] = -100

    # Kiểm tra còn token đầu ra được đánh giá sau khi dịch labels
    if not (labels[:, 1:] != -100).any():
        raise ValueError("Không tìm thấy token mới để đánh giá")

    model.eval()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            labels=labels
        )

        loss = outputs.loss

    perplexity = math.exp(loss.item())

    return perplexity


if __name__ == "__main__":

    # Tải GPT-2 Medium để đánh giá
    eval_model, eval_tokenizer = load_medium_model()

    prompt = "Artificial intelligence is changing the world."

    generated_text = """
    Artificial intelligence helps people solve problems.
    It can support education, healthcare and many other fields.
    """

    ppl = calculate_perplexity(
        eval_model,
        eval_tokenizer,
        prompt,
        generated_text
    )

    print("Perplexity:", ppl)
