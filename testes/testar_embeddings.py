import logging
import os
import sys
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
import logging

load_dotenv()

# ------------------------------
# CONFIGURAÇÃO DE LOG
# ------------------------------
logging.basicConfig(level=logging.INFO,
                    format='(%(asctime)s) %(levelname)s ➧ %(message)s',
                    datefmt='%d-%m-%Y %H:%M:%S')

# ------------------------------
# FUNÇÃO PRINCIPAL
# ------------------------------
def gerar_embeddings_azure(texto: str) -> list:
    """
    Gera embeddings para um texto usando o Azure OpenAI Embeddings.
    Retorna uma lista de floats representando o vetor do texto.
    """
    logging.info("Iniciando geração de embeddings com Azure OpenAI...")
    logging.info("Conectando ao Azure OpenAI Embeddings...")
    try:
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        if not all([azure_deployment, azure_endpoint, api_key, api_version]):
            raise ValueError("⚠️ Variáveis de ambiente do Azure não estão completamente definidas.")

        logging.info("Conectando ao Azure OpenAI Embeddings...")
        embeddings_model = AzureOpenAIEmbeddings(
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )
        logging.info("✅ Cliente de Embeddings conectado com sucesso!")

        # Geração do embedding
        embedding = embeddings_model.embed_query(texto)
        logging.info("✅ Embedding gerado com sucesso!")
        return embedding

    except Exception as e:
        logging.error(f"Erro ao gerar embeddings: {e}")
        sys.exit(1)

logging.info("Testando geração de embeddings...")
try:
    sample_text = "Teste de geração de embeddings com Azure OpenAI."
    embedding = gerar_embeddings_azure(sample_text)
    logging.info("✅ Embeddings gerados com sucesso!")
    logging.info(f"Tipo do resultado: {type(embedding)}")
    logging.info(f"Dimensões do vetor: {len(embedding)}")
    logging.info(f"Primeiros 10 números do vetor: {embedding[:10]}")

except Exception as e:
    logging.error("Erro ao gerar embeddings.")
    logging.error(f"Detalhe do erro: {e}")

finally:
    logging.info("Teste de embeddings concluído.")