import streamlit as st
from transformers.pipelines import pipeline

@st.cache_resource(show_spinner="Carregando modelo…")
def carregar_resumidor():
    pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
    return SummarizerWrapper(pipe)

class SummarizerWrapper:
    def __init__(self, pipe):
        self.pipe = pipe
        self.tokenizer = pipe.tokenizer

    def __getattr__(self, name):
        # Expõe atributos do pipeline original (tokenizer, model, etc.)
        return getattr(self.pipe, name)

    def __call__(self, inputs, max_length=180, min_length=60, do_sample=False, **kwargs):
        if max_length < min_length:
            raise ValueError(
                f"max_length ({max_length}) must be greater than or equal to min_length ({min_length})"
            )

        tok = self.tokenizer

        def clamp_model_max_length():
            model_max = getattr(tok, "model_max_length", 1024)
            # Alguns tokenizers têm valores gigantes; limite a ~1024
            if model_max is None or model_max > 2048:
                model_max = 1024
            return model_max

        model_max = clamp_model_max_length()
        chunk_tokens = min(1000, model_max - 24)

        def token_len(s: str) -> int:
            return len(tok.encode(s, add_special_tokens=False))

        def summarize_one(text: str) -> str:
            if not text or not str(text).strip():
                return ""
            # Curto: sumariza direto com truncation
            if token_len(text) <= chunk_tokens:
                out = self.pipe(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=do_sample,
                    truncation=True,
                    clean_up_tokenization_spaces=True,
                    **kwargs,
                )
                return out[0]["summary_text"]

            # Quebra em chunks por tokens aproximadamente
            words = text.split()
            chunks, cur, cur_len = [], [], 0
            for w in words:
                w_len = len(tok.encode(w, add_special_tokens=False))
                if cur and cur_len + w_len + 1 > chunk_tokens:
                    chunks.append(" ".join(cur))
                    cur, cur_len = [w], w_len
                else:
                    cur.append(w)
                    cur_len += w_len + (1 if cur_len > 0 else 0)
            if cur:
                chunks.append(" ".join(cur))

            partials = []
            for ch in chunks:
                out = self.pipe(
                    ch,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=do_sample,
                    truncation=True,
                    clean_up_tokenization_spaces=True,
                    **kwargs,
                )
                partials.append(out[0]["summary_text"])

            combined = " ".join(partials)
            # Mais uma compressão se ainda ficou grande
            if token_len(combined) > chunk_tokens:
                out = self.pipe(
                    combined,
                    max_length=max_length,
                    min_length=max(30, min_length // 2),
                    do_sample=do_sample,
                    truncation=True,
                    clean_up_tokenization_spaces=True,
                    **kwargs,
                )
                return out[0]["summary_text"]
            return combined

        # Mantém o formato de saída do pipeline: lista de dicts
        if isinstance(inputs, list):
            return [{"summary_text": summarize_one(t)} for t in inputs]
        else:
            return [{"summary_text": summarize_one(inputs)}]


resumidor = carregar_resumidor()
st.title("Resumir texto")
texto = st.text_area("Texto de entrada")
max_length = st.slider("max_length", min_value=10, max_value=500, value=180)
min_length = st.slider("min_length", min_value=5, max_value=max_length, value=60)

if st.button("Resumir"):
    if texto:
        resultado = resumidor(texto, max_length=max_length, min_length=min_length)
        st.text_area("Resumo", resultado[0]["summary_text"], height=200)
