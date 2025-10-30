from dotenv import load_dotenv
import os
import pytest


def test_azure_api_key_loaded():
    """Testa se a chave da API do Azure foi carregada corretamente."""
    load_dotenv()
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    assert api_key is not None, "❌ ERRO: Chave da API do Azure não encontrada."
    assert len(api_key) > 0, "❌ ERRO: Chave da API do Azure está vazia."
    print("✅ SUCESSO: Chave da API do Azure carregada com sucesso.")


def test_azure_endpoint_loaded():
    """Testa se o endpoint do Azure foi carregado corretamente."""
    load_dotenv()
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    if endpoint is None or len(endpoint) == 0:
        pytest.skip("⚠️ ALERTA: Endpoint do Azure não configurado.")
    
    print("✅ SUCESSO: Endpoint do Azure carregado com sucesso.")
