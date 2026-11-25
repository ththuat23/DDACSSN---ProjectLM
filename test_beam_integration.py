from models.model_loader import load_model
from decoding.beam_search import beam_search_decode


def test_beam_with_gpt2():

    # Tải GPT-2 Small
    model, tokenizer = load_model()

    prompt = "Artificial intelligence is"

    result = beam_search_decode(
        model,
        tokenizer,
        prompt,
        max_new_tokens=30,
        beam_width=5,
        length_penalty=0.6,
        min_new_tokens=5
    )

    print("\nPrompt:", prompt)
    print("Generated text:", result)

    assert isinstance(result, str)
    assert len(result.strip()) > 0