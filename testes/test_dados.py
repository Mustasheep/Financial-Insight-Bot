import glob
import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='(%(asctime)s) %(levelname)s ➧ %(message)s',
                    datefmt='%d-%m-%Y %H:%M:%S')

DATA_PATH = "./dados_rpm/"

# Percorrer todos os PDFs do diretório
PDF_PATTERN = os.path.join(DATA_PATH, "*.pdf")
logging.info(f"🔎 Procurando arquivos PDF em: {PDF_PATTERN}")

# Lista todos os arquivos PDF no diretório especificado
pdf_files = glob.glob(PDF_PATTERN)

if not pdf_files:
    logging.warning("Nenhum arquivo PDF encontrado no diretório especificado.")
else:
    logging.info(f"{len(pdf_files)} arquivos PDF encontrados.")
