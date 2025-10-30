import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import pytest


def test_vectorstore_exists_for_bot():
    """Testa se o vetorstore existe para o bot RAG."""
    VECTORSTORE_PATH = "faiss_index"

    assert os.path.exists(VECTORSTORE_PATH),         f"❌ ERRO: Vetorstore não encontrado em '{VECTORSTORE_PATH}'."

    print(f"✅ SUCESSO: Vetorstore encontrado para o bot RAG.")


def test_azure_client_module():
    """Testa se o módulo azure_client pode ser importado."""
    try:
        from src.utils.azure_client import get_azure_embeddings, get_azure_slm
        print("✅ SUCESSO: Módulo azure_client importado com sucesso.")
    except ImportError as e:
        pytest.skip(f"⚠️ ALERTA: Não foi possível importar azure_client: {str(e)}")


def test_rag_chain_creation():
    """Testa a criação da cadeia RAG."""
    load_dotenv()
    VECTORSTORE_PATH = "faiss_index"

    if not os.path.exists(VECTORSTORE_PATH):
        pytest.skip("⚠️ ALERTA: Vetorstore não encontrado.")

    try:
        from src.utils.azure_client import get_azure_embeddings, get_azure_slm
    except ImportError:
        pytest.skip("⚠️ ALERTA: Módulo azure_client não disponível.")

    try:
        embeddings_model = get_azure_embeddings()
        llm = get_azure_slm()

        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings_model,
            allow_dangerous_deserialization=True
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        template = """Você é um assistente especializado em análise de dados econômicos do Banco Central.
Use o contexto fornecido abaixo para responder à pergunta do usuário de forma clara e objetiva.
Se você não souber a resposta com base no contexto, diga: "Desculpe, não posso te ajudar nisso".
Não invente informações.

Contexo:
{context}

Pergunta: {question}

Resposta:"""

        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        print("✅ SUCESSO: Cadeia RAG criada com sucesso.")

    except Exception as e:
        pytest.fail(f"❌ ERRO: Falha ao criar cadeia RAG: {str(e)}")


def test_rag_bot_response():
    """Testa se o bot RAG consegue gerar uma resposta."""
    load_dotenv()
    VECTORSTORE_PATH = "faiss_index"

    if not os.path.exists(VECTORSTORE_PATH):
        pytest.skip("⚠️ ALERTA: Vetorstore não encontrado.")

    try:
        from src.utils.azure_client import get_azure_embeddings, get_azure_slm
    except ImportError:
        pytest.skip("⚠️ ALERTA: Módulo azure_client não disponível.")

    try:
        embeddings_model = get_azure_embeddings()
        llm = get_azure_slm()

        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings_model,
            allow_dangerous_deserialization=True
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        template = """Você é um assistente especializado em análise de dados econômicos do Banco Central.
Use o contexto fornecido abaixo para responder à pergunta do usuário de forma clara e objetiva.
Se você não souber a resposta com base no contexto, diga: "Desculpe, não posso te ajudar nisso".
Não invente informações.

Contexto:
{context}

Pergunta: {question}

Resposta:"""

        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        pergunta = "Quais são os principais riscos para a estabilidade financeira?"
        resposta = rag_chain.invoke(pergunta)

        assert isinstance(resposta, str), "❌ ERRO: Resposta não é uma string."
        assert len(resposta) > 0, "❌ ERRO: Resposta está vazia."

        print(f"✅ SUCESSO: Bot RAG gerou resposta com sucesso ({len(resposta)} caracteres).")

    except Exception as e:
        pytest.fail(f"❌ ERRO: Falha ao gerar resposta do bot: {str(e)}")
