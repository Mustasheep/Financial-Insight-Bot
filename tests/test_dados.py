import glob
import os
import pytest


def test_pdf_files_exist():
    """Testa se existem arquivos PDF no diretório especificado."""
    DATA_PATH = "./dados_rpm/"
    PDF_PATTERN = os.path.join(DATA_PATH, "*.pdf")
    
    pdf_files = glob.glob(PDF_PATTERN)
    
    assert len(pdf_files) > 0, "Nenhum arquivo PDF encontrado no diretório especificado."
    print(f"✅ SUCESSO: {len(pdf_files)} arquivos PDF encontrados.")


def test_data_directory_exists():
    """Testa se o diretório de dados existe."""
    DATA_PATH = "./dados_rpm/"
    
    assert os.path.exists(DATA_PATH), f"⚠️ ALERTA: Diretório {DATA_PATH} não encontrado."
    print(f"✅ SUCESSO: Diretório {DATA_PATH} existe.")