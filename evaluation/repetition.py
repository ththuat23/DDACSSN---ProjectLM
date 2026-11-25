from evaluation.distinct_n import tokenize_text

def repetition_rate(text, n=3):
    """
    Tính tỷ lệ n-gram lặp lại.
    Mặc định sử dụng trigram.
    Kết quả trả về theo phần trăm.
    """

    if n <= 0:
        raise ValueError("n phải lớn hơn 0")

    tokens = tokenize_text(text)

    if len(tokens) < n:
        return 0.0

    ngrams = [
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]
    repeated_count = len(ngrams) - len(set(ngrams))

    return repeated_count / len(ngrams) * 100

if __name__ == "__main__":

    generated_text = """
    Artificial intelligence is changing the world.
    Artificial intelligence helps people solve problems.
    """
    
    repetition = repetition_rate(
        generated_text,
        n=3
    )

    print("Repetition Rate:", repetition, "%")