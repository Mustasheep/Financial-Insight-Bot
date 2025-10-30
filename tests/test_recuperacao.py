import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pytest


def test_vectorstore_exists():
    """Testa se o vetorstore FAISS existe."""
    VECTORSTORE_PATH = "faiss_index"

    assert os.path.exists(VECTORSTORE_PATH),         f"❌ ERRO: Vetorstore não encontrado em '{VECTORSTORE_PATH}'."

    print(f"✅ SUCESSO: Vetorstore encontrado em '{VECTORSTORE_PATH}'.")


def test_load_vectorstore():
    """Testa o carregamento do vetorstore FAISS."""
    load_dotenv()
    VECTORSTORE_PATH = "faiss_index"

    if not os.path.exists(VECTORSTORE_PATH):
        pytest.skip("⚠️ ALERTA: Vetorstore não encontrado.")

    azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    if not all([azure_deployment, azure_endpoint, api_key, api_version]):
        pytest.skip("⚠️ ALERTA: Variáveis de ambiente do Azure não configuradas.")

    try:
        embeddings_model = AzureOpenAIEmbeddings(
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )

        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings_model,
            allow_dangerous_deserialization=True
        )

        print("✅ SUCESSO: Vetorstore FAISS carregado com sucesso.")

    except Exception as e:
        pytest.fail(f"❌ ERRO: Falha ao carregar vetorstore: {str(e)}")


def test_similarity_search():
    """Testa a busca por similaridade no vetorstore."""
    load_dotenv()
    VECTORSTORE_PATH = "faiss_index"

    if not os.path.exists(VECTORSTORE_PATH):
        pytest.skip("⚠️ ALERTA: Vetorstore não encontrado.")

    azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    if not all([azure_deployment, azure_endpoint, api_key, api_version]):
        pytest.skip("⚠️ ALERTA: Variáveis de ambiente do Azure não configuradas.")

    try:
        embeddings_model = AzureOpenAIEmbeddings(
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )

        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings_model,
            allow_dangerous_deserialization=True
        )

        pergunta_usuario = "Quais são os principais riscos para a estabilidade financeira?"
        chunks_relevantes = vectorstore.similarity_search_with_score(
            query=pergunta_usuario,
            k=3
        )

        assert len(chunks_relevantes) > 0, "❌ ERRO: Nenhum chunk relevante encontrado."

        print(f"✅ SUCESSO: Busca por similaridade concluída. {len(chunks_relevantes)} chunks encontrados.")

    except Exception as e:
        pytest.fail(f"❌ ERRO: Falha na busca por similaridade: {str(e)}")
