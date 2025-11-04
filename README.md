# Financial Insight Bot

## Visão Geral

O **Financial Insight Bot** é uma aplicação de IA conversacional que utiliza técnicas avançadas de RAG (Retrieval-Augmented Generation) para transformar relatórios oficiais do Banco Central do Brasil (BACEN) em conhecimento dinâmico. Ele permite que os usuários façam perguntas e obtenham respostas baseadas diretamente nos conteúdos dos Relatórios de Política Monetária (RPM), usando técnicas de NLP (Processamento de Linguagem Natural) e machine learning em nuvem, tornando a análise econômica acessível, precisa e instantânea.

---

## Principais Ferramentas e Tecnologias

- **Python 3.13+**
- **Streamlit**: Interface web intuitiva e interativa para interação com o bot.
- **LangChain**: Orquestração e criação da cadeia RAG (retrieval-augmented generation).
- **FAISS**: Banco de dados vetorial eficiente para busca semântica.
- **Azure OpenAI**: Embeddings e modelos SLM (`gpt-4o-mini`) conectados via API.
- **dotenv**: Gerenciamento seguro de variáveis de ambiente.
- **PDF e processamento de dados**: Extração e vetorização de textos dos relatórios BACEN.

---

## Estrutura do Projeto

```
FINANCIAL-INSIGHT-BOT/
│
├── .venv/                          # Ambiente virtual Python
│
├── app/                            # Aplicação Streamlit
│   ├── __init__.py
│   └── app.py                      # Interface web do chatbot
│
├── dados_rpm/                      # Dados de entrada (PDFs RPM)
│   ├── RPM_Dez_2024.pdf ...        # Relatórios do BACEN
│
├── faiss_index/                    # Banco de dados vetorial
│   ├── index.faiss
│   └── index.pkl
│
├── src/                            # Código-fonte principal
│   ├── agente/
│   │   └── agente.py               # Cadeia RAG e lógica central
│   ├── pipelines/
│   │   ├── pipeline_ingestao.py    # Ingestão e vetorização dos PDFs
│   │   └── processar_dados.py      # Pré-processamento de dados/texto
│   ├── utils/
│   │   └── azure_client.py         # Conexão com Azure OpenAI
│   └── __init__.py
│
├── testes/                         # Testes automatizados/unidade
│   ├── testar_bot.py
│   ├── testar_embeddings.py ...
│
├── .env.example                    # Exemplo - Variáveis de ambiente (chaves API Azure, etc)
├── .gitignore
├── LICENSE
├── README.md                       # Este arquivo
└── requirements.txt                # Dependências Python
```

---

## Funcionamento do Sistema

### 1. Ingestão & Indexação
- PDFs dos relatórios RPM são adicionados ao diretório `dados_rpm/`.
   - [Relatórios de Política Monetrária - BACEN](https://www.bcb.gov.br/publicacoes/rpm/cronologicos) 
- Rode `pipeline_ingestao.py` para processar, quebrar em textos menores, gerar embeddings e indexar tudo no FAISS.
- Resultado: um banco vetorial consultável e pronto para recuperação semântica.

### 2. Agente RAG
- O código em `src/agente/agente.py` implementa a cadeia RAG, conectando o retriever (FAISS) ao modelo de linguagem `gpt-4o-mini` para gerar respostas fundamentadas apenas nos trechos dos relatórios recuperados.

### 3. Interface Usuário (Chatbot)
- O usuário interage via interface Streamlit (`app/app.py`), podendo perguntar de forma natural sobre inflação, políticas monetárias, projeções do BACEN e muito mais.
- O sistema responde com base nos relatórios, nunca inventando dados.

### 4. Testes Automatizados
- Scripts no diretório `testes/` validam extração, embeddings, recuperação e respostas do bot.

---

## Como Executar o Projeto

### Pré-requisitos
- Python 3.13+
- Conta Azure/OpenAI (chaves nas variáveis de ambiente `.env`)

### Instalação
1. Clone o repositório e entre na pasta raíz.
2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # ou .venv\Scripts\activate no Windows
   ```
3. Instale dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure `.env` com suas chaves Azure/OpenAI conforme o exemplo em `.env.example`.
5. Adicione os PDFs dos relatórios ao diretório `dados_rpm/`.

### Pipeline de Indexação
Execute a ingestão e indexação dos documentos:
```bash
python -m src.pipelines.pipeline_ingestao
```

### Execução do Bot
Inicie a interface:
```bash
streamlit run app/app.py
```

### Testes
```bash
pytest tests/
```

---

## Detalhes sobre Funcionamento

- O bot **nunca responde usando conhecimento externo** – tudo é gerado a partir dos relatórios BACEN.
- Guardrails embutidos impedem alucinações e respostas não fundamentadas.
- Logs detalhados são emitidos para rastreabilidade (ajuda em depuração e operação em produção).

---

## Contato & Licença

Distribuído sob a licença MIT.
Dúvidas técnicas? Abra uma Issue ou Pull Request.
