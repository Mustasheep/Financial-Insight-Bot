import os
import glob
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format='(%(asctime)s) %(levelname)s - %(message)s',
                    datefmt='%d-%m-%Y %H:%M:%S')

logging.info("Iniciando o processamento dos dados...")
DATA_PATH = "./dados_rpm/"
PDF_PATTERN = os.path.join(DATA_PATH, "*.pdf")

# Definindo o tamanho do chunk e o overlap
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400

logging.info(f"Procurando arquivos PDF em: {PDF_PATTERN}")
pdf_files = glob.glob(PDF_PATTERN)

if not pdf_files:
    logging.warning("Nenhum arquivo PDF encontrado.")
    exit()
logging.info(f"{len(pdf_files)} arquivos PDF encontrados.")

logging.info("Iniciando o carregamento e divisão dos documentos...")
all_chunks = []
start_time = time.time()

logging.info("Iniciando o fragmentador de texto...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

total_paginas_processadas = 0

for file_path in pdf_files:
    file_name = os.path.basename(file_path)
    logging.info(f"⏳ Processando arquivo: {file_name}...")

    loader = PDFPlumberLoader(file_path)
    docs = loader.load() 

    docs_filtrados = [doc for doc in docs if len(doc.page_content) > 100]
    total_paginas_processadas += len(docs_filtrados)
    
    chunks = text_splitter.split_documents(docs_filtrados)
    
    all_chunks.extend(chunks)
    
    logging.info(f"Arquivo {file_name} dividido em {len(chunks)} chunks.")
end_time = time.time()
logging.info(f"✅ Processamento concluído em {end_time - start_time:.2f} segundos.")
logging.info(f"➥  Total de PDFs processados: {len(pdf_files)}")
logging.info(f"➥  Total de páginas processadas: {total_paginas_processadas}")
logging.info(f"➥  Total de chunks gerados: {len(all_chunks)}")

if all_chunks:
    print("\n--- Exemplo de Chunk (Chunk #200) ---", flush=True)
    chunk_exemplo = all_chunks[200]
    
    print(f"Conteúdo (primeiros 300 caracteres):\n\"...{chunk_exemplo.page_content[:300]}...\"\n", flush=True)
    
    print(f"📋 Metadados:\n{chunk_exemplo.metadata}\n", flush=True)