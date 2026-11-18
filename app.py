import streamlit as st

from models.model_loader import load_model
from decoding.greedy import greedy_decode


@st.cache_resource
def get_model():
    return load_model()


model, tokenizer = get_model()
st.title("Language Model Decoding")
st.write("Decoding Strategies Project")

prompt = st.text_area(
    "Nhập văn bản đầu vào:",
    "Artificial intelligence is"
)

max_tokens = st.slider(
    "Maximum new tokens:",
    min_value=10,
    max_value=100,
    value=50
)

method = st.selectbox(
    "Chọn phương pháp giải mã:",
    [
        "Greedy Search",
        "Beam Search",
        "Top-k Sampling",
        "Top-p Sampling",
        "Temperature Sampling"
    ]
)

if st.button("Generate"):
    if method == "Greedy Search":
        result = greedy_decode(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_tokens
        )
        st.subheader("Kết quả:")
        st.write(result)
