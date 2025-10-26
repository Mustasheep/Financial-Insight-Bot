import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

def get_azure_embeddings():
    """Retorna o cliente AzureOpenAIEmbeddings pronto para uso."""
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )
