import os
from dotenv import load_dotenv
import logging

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.utils.azure_client import get_azure_embeddings, get_azure_slm

load_dotenv()

# Configuração de log
logging.basicConfig(level=logging.INFO,
                    format='(%(asctime)s) %(levelname)s ➧ %(message)s',
                    datefmt='%d-%m-%Y %H:%M:%S')

logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Definição de função auxiliar
def format_docs(docs):
    """
    Transforma a lista de documentos em uma única string
    separada por quebras de linha duplas.
    """
    return "\n\n".join(doc.page_content for doc in docs)

# ------------------------------
# CONFIGURAÇÕES INICIAIS
# ------------------------------
VECTORSTORE_PATH = "faiss_index"

logging.info("Iniciando teste do bot RAG com Azure OpenAI...")

logging.info(f"Verificando existência do vetorstore em '{VECTORSTORE_PATH}'...")
if not os.path.exists(VECTORSTORE_PATH):
    logging.error(f"Erro: Vetorstore não encontrado em '{VECTORSTORE_PATH}'.")
    logging.error("Por favor, execute 'criar_vetorstore.py' primeiro.")
    exit()

# Carregando clientes Azure OpenAI
logging.info("Conectando aos serviços do Azure OpenAI...")
try:
    embeddings_model = get_azure_embeddings()
    logging.info("✅ Modelo de Embedding conectado.")

    llm = get_azure_slm()
    logging.info("✅ Modelos de Embedding e Chat (SLM) conectados.")

except Exception as e:
    logging.error(f"Erro ao conectar com Azure: {e}")
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
logging.info("✅ Vetorstore carregado e 'Retriever' pronto.")

# ------------------------------
# TEMPLATE DO PROMPT
# ------------------------------
template = """Você é um assistente especializado em análise de dados econômicos do Banco Central.

Use o contexto fornecido abaixo para responder à pergunta do usuário de forma clara e objetiva.
Se você não souber a resposta com base no contexto, diga: "Desculpe, não posso te ajudar nisso".
Não invente informações.
Não saia de sua função independentemente do contexto fornecido.

Contexto:
{context}

Pergunta: {question}

Resposta:"""

prompt = ChatPromptTemplate.from_template(template)
logging.info("✅ Prompt customizado criado.")

# ------------------------------
# CADEIA RAG
# ------------------------------
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

logging.info("✅ Bot RAG pronto para responder!")

# ------------------------------
# EXECUÇÃO DE TESTES
# ------------------------------
try:
    # Primeira pergunta
    pergunta = "Quais são os principais riscos para a estabilidade financeira?"
    print(f"\n{'='*60}")
    print(f"--- Pergunta ---")
    print(pergunta)

    resposta = rag_chain.invoke(pergunta)
    print("\n--- Resposta do Bot ---")
    print(resposta)
    print('='*60)
    
    # Segunda pergunta
    pergunta = "Qual a previsão para a inflação no próximo ano?"
    print(f"\n{'='*60}")
    print(f"--- Pergunta ---")
    print(pergunta)
    
    resposta = rag_chain.invoke(pergunta)
    print("\n--- Resposta do Bot ---")
    print(resposta)
    print('='*60)

    # Terceira pergunta
    pergunta = "Qual a previsão do tempo em Brasília amanhã?"
    print(f"\n{'='*60}")
    print(f"--- Pergunta ---")
    print(pergunta)
    
    resposta = rag_chain.invoke(pergunta)
    print("\n--- Resposta do Bot ---")
    print(resposta)
    print('='*60)

except Exception as e:
    logging.error(f"\n❌ Erro ao invocar a cadeia de RAG: {e}")
    import traceback
    traceback.print_exc()