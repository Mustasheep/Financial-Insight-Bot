import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
import logging

logging.basicConfig(level=logging.INFO,
                    format='(%(asctime)s) %(levelname)s ➧ %(message)s',
                    datefmt='%d-%m-%Y %H:%M:%S')

logging.info("Carregando variáveis de ambiente...")
load_dotenv()

logging.info("Procurando arquivos PDF no diretório especificado...")
DATA_PATH = "./dados_rpm/"
PDF_PATTERN = os.path.join(DATA_PATH, "*.pdf")
pdf_files = glob.glob(PDF_PATTERN)

if not pdf_files:
    logging.warning("Nenhum arquivo PDF encontrado no diretório especificado.")
else:
    logging.info(f"{len(pdf_files)} arquivos PDF encontrados. Iniciando o carregamento...")

    primeiro_pdf = pdf_files[0]
    logging.info(f"Iniciando extração de texto do arquivo: {os.path.basename(primeiro_pdf)}...")

    # Loader para o caminho do arquivo
    loader = PDFPlumberLoader(primeiro_pdf)
    
    # Criando um documento a partir do PDF
    docs = loader.load()

    logging.info(f"Extração concluída. Número de páginas extraídas: {len(docs)}")
    if len(docs) > 10:
        pagina_10 = docs[9]
        logging.info("Mostrando o conteúdo da página 10...\n")
        # Exibindo as primeiras 1000 caracteres da página 10
        print(f"\"...{pagina_10.page_content[:1000]}...\"\n", flush=True)

        logging.info("Exibindo metadados da página 10...\n")
        print(f"{pagina_10.metadata}\n", flush=True)