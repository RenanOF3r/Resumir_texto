import streamlit as st
from transformers import pipeline

@st.cache_resource
def carregar_resumidor():
    return pipeline('summarization', model='facebook/bart-large-cnn')

resumidor = carregar_resumidor()

def resumir_texto(texto, max_length=300, min_length=100):
    resumo = resumidor(texto, max_length=max_length, min_length=min_length, do_sample=False)
    return resumo[0]['summary_text']

st.title("Resumidor de Textos")

texto_entrada = st.text_area("Cole o texto para resumir", height=200)

if st.button("Resumir"):
    if not texto_entrada.strip():
        st.warning("Por favor, insira um texto para resumir.")
    else:
        with st.spinner("Resumindo..."):
            try:
                resumo = resumir_texto(texto_entrada)
                st.subheader("Resumo gerado:")
                st.write(resumo)
            except Exception as e:
                st.error(f"Erro ao resumir o texto: {e}")
