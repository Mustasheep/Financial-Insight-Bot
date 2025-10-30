import os
import logging
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.utils.azure_client import get_azure_embeddings


load_dotenv()

VECTORSTORE_PATH = "faiss_index"

# ------------------------------
# CONFIGURAÇÃO DE LOG
# ------------------------------
logging.basicConfig(level=logging.INFO,
                    format='(%(asctime)s) %(levelname)s ➧ %(message)s',
                    datefmt='%d-%m-%Y %H:%M:%S')

# ------------------------------
# CARREGANDO MODELOS E VETORSTORE
# ------------------------------
logging.info("Estabelecendo conexão com Azure Embeddings...")

try:
    embeddings_model = get_azure_embeddings()
    logging.info("✅ Conexão com Azure Embeddings bem-sucedida.")
except Exception as e:
    logging.error(f"Erro ao conectar com Azure Embeddings: {e}")
    exit()

logging.info("Carregando ao Vetorstore FAISS...")

try:
    vectorstore = FAISS.load_local(VECTORSTORE_PATH,
                                   embeddings_model,
                                   allow_dangerous_deserialization=True
                                   )
    logging.info("✅ Vetorstore FAISS carregado com sucesso.")
except Exception as e:
    logging.error(f"Erro ao carregar o Vetorstore FAISS: {e}")
    exit()

# ------------------------------
# TESTE DE RECUPERAÇÃO DE DOCUMENTOS
# ------------------------------
logging.info("Iniciando o teste de recuperação de documentos...")

pergunta_usuario = "Quais são os principais riscos para a estabilidade financeira?"
logging.info(f"📋 Pergunta do usuário: {pergunta_usuario}")

logging.info("Realizando busca por similaridade no Vetorstore FAISS...")

try:
    chunks_relevantes = vectorstore.similarity_search_with_score(
        query=pergunta_usuario, 
        k=3
    )
    logging.info(f"✅ Recuperação concluída. Encontrados {len(chunks_relevantes)} chunks:")

    # Inspecionando resultados
    for i, (chunk, score) in enumerate(chunks_relevantes):
        print(f"\n--- CHUNK RELEVANTE #{i+1} | Score: {score:.4f}---", flush=True)
        
        print(f"Fonte: {chunk.metadata.get('source', 'N/A')}", flush=True)
        print(f"Página: {chunk.metadata.get('page', 'N/A')}", flush=True)
        
        print("\nConteúdo:", flush=True)
        print(chunk.page_content, flush=True)
        
except Exception as e:
    logging.error(f"Erro durante a busca por similaridade: {e}")

finally:
    logging.info("Teste de recuperação de documentos concluído.")