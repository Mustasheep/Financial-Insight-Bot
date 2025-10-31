import os
import sys
import time
import logging
from langchain_community.vectorstores import FAISS

# Importando módulos
from src.utils.azure_client import get_azure_embeddings
from src.pipelines.processar_dados import processar_dados
from src.utils.setup_log import setup_logging

setup_logging()

# ------------------------------
# FUNÇÃO PRINCIPAL
# ------------------------------
def pipeline_ingestao():
    """
    Pipeline de ingestão de dados que processa PDFs, gera embeddings e armazena em FAISS.
    """
    logging.info("Iniciando o pipeline de ingestão de dados...")
    start_time = time.time()

    # Carregar e fragmentar PDFs
    DATA_PATH = "./dados_rpm/"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 250
    BATCH_SIZE = 25
    VECTORSTORE_PATH = os.path.join("faiss_index")
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)

    chunks = processar_dados(DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        logging.error("Nenhum chunk foi gerado. Encerrando...")
        sys.exit(1)

    # Conectar ao Azure Embeddings
    embeddings_model = get_azure_embeddings()

    # Gerar embeddings em batch
    lista_de_textos = [chunk.page_content for chunk in chunks]
    embeddings = []

    logging.info(f"Gerando embeddings em batches de {BATCH_SIZE} chunks...")

    for i in range(0, len(lista_de_textos), BATCH_SIZE):
        batch = lista_de_textos[i:i + BATCH_SIZE]
        try:
            batch_embeddings = embeddings_model.embed_documents(batch)
            embeddings.extend(batch_embeddings)
            logging.info(f"✅ Batch {i // BATCH_SIZE + 1} concluído ({len(batch_embeddings)} embeddings).")
        except Exception as e:
            logging.warning(f"⚠️ Erro no batch {i // BATCH_SIZE + 1}: {e}")
            time.sleep(60)

    # Criar índice FAISS
    logging.info("Iniciando criação do índice FAISS...")

    vector_store = FAISS.from_embeddings(
        text_embeddings=list(zip(lista_de_textos, embeddings)),
        embedding=embeddings_model,
        metadatas=[chunk.metadata for chunk in chunks]
    )

    vector_store.save_local(VECTORSTORE_PATH)
    elapsed = time.time() - start_time

    logging.info(f"Vetorstore criado e salvo em '{VECTORSTORE_PATH}'.")
    logging.info(f"✅ Pipeline de ingestão concluído com sucesso em {elapsed:.2f} segundos.")

if __name__ == "__main__":
    pipeline_ingestao()
