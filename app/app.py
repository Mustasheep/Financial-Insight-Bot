import streamlit as st
import os
import sys
import logging
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

# Caminho do módulo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.agente.agente import create_rag_chain
from src.utils.setup_log import setup_logging

setup_logging()

# ------------------------------
# STREAMLIT CONFIG
# ------------------------------
st.set_page_config(
    page_title="Monetary Insight Bot",
    page_icon="📊",
    layout="wide")

# ------------------------------
# CABEÇALHO PRINCIPAL
# ------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f9fafc;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 0rem;
    }
    </style>
""", unsafe_allow_html=True)

colored_header(
    label="🤖 Monetary Insight Bot",
    description="Seu assistente de IA para analisar relatórios do Banco Central",
    color_name="blue-70",
)
st.markdown("Use o poder do RAG para responder perguntas com base nos **Relatórios de Política Monetária (RPM)** mais recentes.")
st.divider()

# ------------------------------
# SIDEBAR
# ------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://images.icon-icons.com/3729/PNG/512/salary_marketing_income_financial_earn_money_is_work_icon_230565.png" width="120">
        </div>
        """,
        unsafe_allow_html=True)
    st.markdown("---")
    st.header("Sobre o Projeto")
    st.info("""
    Este projeto foi desenvolvido por [**Thiago de Assis**](https://www.linkedin.com/in/thiago-mustasheep/)  
    para demonstrar o uso de **RAG com SLMs** em análises financeiras.""")
    st.markdown("---")
    st.markdown("""
    **Como usar:**
    1. Digite uma pergunta na caixa ao lado.  
    2. Aguarde enquanto o modelo busca respostas nos relatórios.  
    3. Veja a resposta e as referências encontradas.""")
    add_vertical_space(3)
    st.caption("Versão 1.2 • Powered by Streamlit & Azure OpenAI")

# ------------------------------
# CARREGAR RAG
# ------------------------------
@st.cache_resource
def load_rag_chain():
    logging.info("Iniciando cache: Carregando pipeline RAG...")
    try:
        chain = create_rag_chain()
        logging.info("Pipeline RAG carregado com sucesso.")
        logging.info("Bot RAG pronto para responder!")
        return chain
    except FileNotFoundError as e:
        st.error(f"Erro ao carregar o pipeline RAG: {e}")
        st.info("Vetorstore não encontrado. Execute o pipeline de ingestão primeiro.")
        return None
    except Exception as e:
        st.error(f"Erro inesperado ao carregar o pipeline RAG: {e}")
        return None

rag_chain = load_rag_chain()

# ------------------------------
# ÁREA PRINCIPAL
# ------------------------------
if rag_chain:
    st.markdown("### Faça sua pergunta")
    user_question = st.text_input(
        "Digite aqui sua pergunta:",
        placeholder="Ex: Qual a projeção do IPCA para 2025?",
        label_visibility="collapsed"
    )

    ask_col, reset_col = st.columns([4, 1])
    with ask_col:
        ask_button = st.button("Perguntar", use_container_width=True)
    with reset_col:
        clear_button = st.button("Limpar", use_container_width=True)

    if ask_button:
        if user_question:
            with stylable_container("resposta_box", css_styles="""
                { background-color: white; border-radius: 12px; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.08); }
            """):
                st.subheader("📃 Resposta")
                with st.spinner("Analisando os relatórios... ⏳"):
                    resposta = rag_chain.invoke(user_question)

                st.markdown(resposta["result"])

            st.markdown("### 📚 Referências")
            palavras_chave_falha = ["não encontrei", "não há informações"]

            if not any(p in resposta["result"].lower() for p in palavras_chave_falha):
                for i, doc in enumerate(resposta["source_documents"], 1):
                    source_name = os.path.basename(doc.metadata.get("source", "Desconhecido"))
                    pagina = doc.metadata.get("page", "?")
                    with st.expander(f"Referência {i}: {source_name} (pág. {pagina})"):
                        st.write(doc.page_content)
            else:
                st.info("Nenhuma referência relevante encontrada para esta resposta.")
        else:
            st.warning("Por favor, digite uma pergunta.")
    elif clear_button:
        st.rerun()
else:
    st.error("O aplicativo não pôde ser iniciado. Verifique os logs.")
