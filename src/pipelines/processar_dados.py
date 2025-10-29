from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import glob
import time
import logging
import sys

# ------------------------------
# CONFIGURAÇÃO DE LOG
# ------------------------------
logging.basicConfig(level=logging.INFO,
                    format='(%(asctime)s) %(levelname)s ➧ %(message)s',
                    datefmt='%d-%m-%Y %H:%M:%S')

# ------------------------------
# FUNÇÃO PRINCIPAL
# ------------------------------
def processar_dados(
    data_path: str = "./dados_rpm/",
    chunk_size: int = 1500,
    chunk_overlap: int = 200
):
    """
    Carrega todos os PDFs do diretório especificado, divide os textos em chunks e retorna uma lista de documentos processados.

    Args:
        data_path (str): Caminho para o diretório contendo os PDFs.
        chunk_size (int): Tamanho máximo de cada chunk.
        chunk_overlap (int): Número de caracteres sobrepostos entre os chunks.

    Returns:
        list: Lista contendo os chunks processados.
    """
    logging.info("Processamento dos dados...")
    pdf_pattern = os.path.join(data_path, "*.pdf")
    logging.info(f"Procurando arquivos PDF em: {pdf_pattern}")
    pdf_files = glob.glob(pdf_pattern)

    if not pdf_files:
        logging.warning("Nenhum arquivo PDF encontrado. Encerrando o processamento...")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    all_chunks = []
    total_paginas = 0

    for file_path in pdf_files:
        file_name = os.path.basename(file_path)
        logging.info(f"⏳ Processando arquivo: {file_name}")

        try:
            loader = PDFPlumberLoader(file_path)
            docs = loader.load()

            # Remover páginas muito curtas
            docs_filtrados = [d for d in docs if len(d.page_content) > 100]
            total_paginas += len(docs_filtrados)

            chunks = text_splitter.split_documents(docs_filtrados)
            all_chunks.extend(chunks)

            logging.info(f"{file_name} dividido em {len(chunks)} chunks.")

        except Exception as e:
            logging.error(f"Erro ao processar {file_name}: {e}")

    logging.info(f"Total de PDFs: {len(pdf_files)} | Páginas: {total_paginas} | Chunks: {len(all_chunks)}")
    return all_chunks
