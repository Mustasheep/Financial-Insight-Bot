import os
import glob
from langchain_community.document_loaders import PDFPlumberLoader
import pytest


def test_pdf_extraction():
    """Testa a extração de texto de arquivos PDF."""
    DATA_PATH = "./dados_rpm/"
    PDF_PATTERN = os.path.join(DATA_PATH, "*.pdf")

    pdf_files = glob.glob(PDF_PATTERN)

    if not pdf_files:
        pytest.skip("⚠️ ALERTA: Nenhum arquivo PDF encontrado para extração.")

    primeiro_pdf = pdf_files[0]

    try:
        loader = PDFPlumberLoader(primeiro_pdf)
        docs = loader.load()

        assert len(docs) > 0, "❌ ERRO: Nenhuma página foi extraída do PDF."
        print(f"✅ SUCESSO: Extração concluída. {len(docs)} páginas extraídas do arquivo {os.path.basename(primeiro_pdf)}.")

    except Exception as e:
        pytest.fail(f"❌ ERRO: Falha na extração de PDF: {str(e)}")


def test_pdf_content_not_empty():
    """Testa se o conteúdo extraído não está vazio."""
    DATA_PATH = "./dados_rpm/"
    PDF_PATTERN = os.path.join(DATA_PATH, "*.pdf")

    pdf_files = glob.glob(PDF_PATTERN)

    if not pdf_files:
        pytest.skip("⚠️ ALERTA: Nenhum arquivo PDF encontrado.")

    primeiro_pdf = pdf_files[0]

    try:
        loader = PDFPlumberLoader(primeiro_pdf)
        docs = loader.load()

        if len(docs) > 0:
            assert len(docs[0].page_content) > 0, "❌ ERRO: Conteúdo da primeira página está vazio."
            print(f"✅ SUCESSO: Conteúdo extraído não está vazio ({len(docs[0].page_content)} caracteres).")

    except Exception as e:
        pytest.fail(f"❌ ERRO: {str(e)}")
