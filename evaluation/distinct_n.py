import re

def tokenize_text(text):
    """Tách văn bản thành các token đơn giản."""
    return re.findall(r"\w+|[^\w\s]", text.lower())

def distinct_n(text, n=1):

    if n <= 0:
        raise ValueError("n phải lớn hơn 0")

    tokens = tokenize_text(text)

    if len(tokens) < n:
        return 0.0

    ngrams = [
        tuple(tokens[i:i+n])
        for i in range(len(tokens) - n + 1)
    ]

    return len(set(ngrams)) / len(ngrams)

def calculate_distinct_scores(text):

    return {
        "distinct_1": distinct_n(text, 1),
        "distinct_2": distinct_n(text, 2),
        "distinct_3": distinct_n(text, 3)
    }

if __name__ == "__main__":

    generated_text = """
    Artificial intelligence is changing the world.
    Artificial intelligence helps people solve problems.
    """

    scores = calculate_distinct_scores(generated_text)

    print("Distinct-1:", scores["distinct_1"])
    print("Distinct-2:", scores["distinct_2"])
    print("Distinct-3:", scores["distinct_3"])
