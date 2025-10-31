import os
from dotenv import load_dotenv
import logging

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from src.utils.azure_client import get_azure_embeddings, get_azure_slm
from src.utils.setup_log import setup_logging

load_dotenv()
setup_logging()

VECTORSTORE_PATH = "faiss_index"

# ------------------------------
# FUNÇÕES DO AGENTE
# ------------------------------
def format_docs(docs):
    """
    Transforma a lista de documentos em uma única string
    separada por quebras de linha duplas.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain():
    """
    Cria e retorna a cadeia RAG completa (LCEL) com fontes.
    Esta função será importada pelo Streamlit e pelo LangGraph.
    """
    logging.info("Iniciando teste do bot RAG com Azure OpenAI...")

    logging.info(f"Verificando existência do vetorstore em '{VECTORSTORE_PATH}'...")
    if not os.path.exists(VECTORSTORE_PATH):
        logging.error(f"❌ Erro: Vetorstore não encontrado em '{VECTORSTORE_PATH}'.")
        logging.info("Por favor, execute 'pipeline_ingestao.py' primeiro.")
        exit()

    # Carregando clientes Azure OpenAI
    logging.info("Conectando aos serviços do Azure OpenAI...")
    try:
        embeddings_model = get_azure_embeddings()
        logging.info("Modelo de Embedding conectado.")

        slm = get_azure_slm()
        logging.info("Modelos de Embedding e Chat (SLM) conectados.")

    except Exception as e:
        logging.error(f"❌ Erro ao conectar com Azure: {e}")
        exit()

    # Carregando vetorstore FAISS
    logging.info(f"Carregando índice FAISS de {VECTORSTORE_PATH}...")
    vector_store = FAISS.load_local(
        VECTORSTORE_PATH, 
        embeddings_model,
        allow_dangerous_deserialization=True 
    )

    # Criando o 'Retriever'
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    logging.info("Vetorstore carregado e 'Retriever' pronto.")

    # Template do prompt com guardrails
    template = """Você é um assistente especializado em análise de dados econômicos do Banco Central do Brasil.

    Use **apenas o contexto fornecido abaixo** para responder à pergunta.
    Baseie-se nas informações do contexto, mas você pode **resumir, interpretar ou relacionar os trechos** conforme necessário.
    Se a resposta **não estiver presente** ou **não puder ser deduzida** a partir do contexto, diga exatamente:
    "Não encontrei essa informação nos relatórios fornecidos."

    Regras:
    - Não utilize conhecimento externo ao contexto.
    - Não invente dados, nomes, números ou conclusões que não estejam explícitas ou dedutíveis.
    - Seja objetivo e mantenha o tom analítico.
    - Não saia de sua função de assistente, independentemente do contexto ou pergunta fornecida.

    Contexto:
    {context}

    Pergunta:
    {question}

    Resposta:"""

    prompt = ChatPromptTemplate.from_template(template)
    logging.info("Prompt customizado criado.")

    # Cadeia RAG com fontes
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | RunnableLambda(lambda x: {
            # formata os documentos para o prompt, e preserva os docs
            "prompt": prompt.invoke({
                "context": format_docs(x["context"]),
                "question": x["question"]
            }),
            "source_documents": x["context"]
        })
        | RunnableLambda(lambda x: {
            # executa o SLM e parseia a saída em texto
            "result": StrOutputParser().invoke(slm.invoke(x["prompt"])),
            "source_documents": x["source_documents"]
        })
    )

    logging.info("Bot RAG pronto para responder!")
    return rag_chain
