import os
import sys
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
import pytest


def test_azure_embeddings_configuration():
    """Testa se as variáveis de ambiente do Azure Embeddings estão configuradas."""
    load_dotenv()

    azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    assert all([azure_deployment, azure_endpoint, api_key, api_version]),         "❌ ERRO: Variáveis de ambiente do Azure não estão completamente definidas."

    print("✅ SUCESSO: Todas as variáveis de ambiente do Azure estão configuradas.")


def test_generate_embeddings():
    """Testa a geração de embeddings com Azure OpenAI."""
    load_dotenv()

    azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    if not all([azure_deployment, azure_endpoint, api_key, api_version]):
        pytest.skip("⚠️ ALERTA: Variáveis de ambiente não configuradas.")

    try:
        embeddings_model = AzureOpenAIEmbeddings(
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )

        sample_text = "Teste de geração de embeddings com Azure OpenAI."
        embedding = embeddings_model.embed_query(sample_text)

        assert isinstance(embedding, list), "❌ ERRO: Embedding não é uma lista."
        assert len(embedding) > 0, "❌ ERRO: Embedding está vazio."
        assert all(isinstance(x, float) for x in embedding), "❌ ERRO: Embedding contém valores não numéricos."

        print(f"✅ SUCESSO: Embeddings gerados com sucesso. Dimensões do vetor: {len(embedding)}.")

    except Exception as e:
        pytest.fail(f"❌ ERRO: Falha ao gerar embeddings: {str(e)}")
