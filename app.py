import streamlit as st
from transformers import pipeline

# Inicializar pipelines (carregados uma vez)
@st.cache_resource
def carregar_modelos():
    tradutor = pipeline('translation', model='Helsinki-NLP/opus-mt-mul-pt')
    resumidor = pipeline('summarization', model='facebook/bart-large-cnn')
    return tradutor, resumidor

tradutor, resumidor = carregar_modelos()

def traduzir_para_portugues(texto):
    traducao = tradutor(texto)
    return traducao[0]['translation_text']

def resumir_texto(texto, max_length=130, min_length=30):
    resumo = resumidor(texto, max_length=max_length, min_length=min_length, do_sample=False)
    return resumo[0]['summary_text']

def traduzir_e_resumir(texto_original):
    texto_em_portugues = traduzir_para_portugues(texto_original)
    resumo = resumir_texto(texto_em_portugues)
    return resumo

# Interface Streamlit
st.title("Resumidor de Textos - Amanda")

st.write("Cole o texto que deseja resumir (em qualquer idioma) e clique em 'Resumir'.")

texto_entrada = st.text_area("Texto original", height=200)

if st.button("Resumir"):
    if not texto_entrada.strip():
        st.warning("Por favor, insira um texto para resumir.")
    else:
        with st.spinner("Processando..."):
            try:
                resumo_gerado = traduzir_e_resumir(texto_entrada)
                st.subheader("Resumo gerado:")
                st.write(resumo_gerado)
            except Exception as e:
                st.error(f"Erro ao processar o texto: {e}")
