from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_model(model_name="gpt2"):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Tải tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Tải mô hình
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Chuyển model sang CPU hoặc GPU
    model.to(device)
    # Chuyển sang chế độ đánh giá
    model.eval()

    return model, tokenizer

def load_medium_model(model_name="openai-community/gpt2-medium"):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Tải tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Tải mô hình
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.to(device)
    model.eval()

    return model, tokenizer