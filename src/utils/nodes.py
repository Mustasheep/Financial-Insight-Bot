from typing import List, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from src.agente.agente import create_rag_chain

class AgentState(TypedDict):
    question: str 
    answer: str 
    source_documents: List[dict] 
    chat_history: List[BaseMessage]

def node_executar_rag(state: AgentState):
    """
    Nó para executar a cadeia RAG
    """
    pergunta = state["question"]
    rag_chain = create_rag_chain()
    resposta = rag_chain.invoke(pergunta)

    return {
        "answer": resposta["answer"],
        "source_documents": resposta["context"]
    }

def node_atualizar_memoria(state: AgentState):
    """
    Adiciona a pergunta e a resposta ao histórico.
    """
    history = state.get("chat_history", [])
    
    # Adiciona a pergunta do humano e a resposta da IA
    history.append(HumanMessage(content=state["question"]))
    history.append(AIMessage(content=state["answer"]))
    
    return {"chat_history": history}