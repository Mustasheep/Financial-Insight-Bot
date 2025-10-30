from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='(%(asctime)s) %(levelname)s ➧ %(message)s',
                    datefmt='%d-%m-%Y %H:%M:%S')

load_dotenv() 

api_key = os.getenv("AZURE_OPENAI_API_KEY")

logging.info("🔎 Verificando a chave da API do Azure...")
try:
    if api_key:
        logging.info("Chave da API do Azure carregada com sucesso.")
except Exception as e:
    logging.error(f"Erro ao carregar a chave da API do Azure: {e}")