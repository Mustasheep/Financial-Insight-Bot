# Financial Insight Bot

## Contexto 

Os relatórios do BACEN giram em torno da política monetária, que é o conjunto de ações que o Banco Central toma para controlar a quantidade de dinheiro na economia e, assim, manter a inflação sob controle.

**Problema:** Relatórios financeiros, como o Relatório Trimestral de Inflação (RTI) ou o Relatório de Política Monetária (RPM) do Banco Central (BACEN), são documentos importantes. Eles ditam políticas, movem mercados e afetam a vida de todos. No entanto, eles são longos, densos e cheios de jargões técnicos. Encontrar uma informação específica exige leitura manual demorada.

**Solução:** O Financial Insight Bot é um agente de IA especialista. O objetivo é transformar esses documentos estáticos (PDFs) em uma base de conhecimento dinâmica e conversacional.

## Funcionamento

**1.** Ingestão de Dados: Armazenar o sistema com os PDFs do BACEN.

**2.** Indexação (O RAG): Quebrar os textos, transformá-los em vetores (embeddings) e armazená-los em um banco de dados vetorial (Vetorstore).

**3.** Recuperação e Geração: O sistema primeiro buscará os trechos mais relevantes no Vetorstore (Recuperação) e, em seguida, usará um LLM para gerar uma resposta coesa usando apenas aqueles trechos (Geração).

---

## Jargões Essenciais de Inflação e Juros

- **Inflação:** É o aumento generalizado e contínuo dos preços de bens e serviços. Se a inflação está alta, o dinheiro compra menos coisas.

- **IPCA** (Índice Nacional de Preços ao Consumidor Amplo): Este é o índice oficial de inflação do Brasil, medido pelo IBGE. Quando o BACEN fala em "meta de inflação", ele está se referindo à meta para o IPCA.

- **Taxa Selic** (Sistema Especial de Liquidação e de Custódia): Esta é a taxa básica de juros da economia brasileira. É a principal ferramenta do BACEN para controlar a inflação.

  - Se a inflação está alta, o BACEN aumenta a Selic. Isso torna o crédito mais caro, desestimulando o consumo e o investimento, o que (em tese) "esfria" a economia e ajuda a baixar os preços. Se a inflação está baixa ou controlada, o BACEN pode diminuir a Selic para estimular a atividade econômica.

- **Copom** (Comitê de Política Monetária): É o órgão do BACEN que, a cada 45 dias, se reúne para decidir qual será a meta da Taxa Selic. Nossos relatórios (RPM) são publicados pelo Copom para justificar essas decisões.

## Jargões de Atividade Econômica

- **PIB** (Produto Interno Bruto): É a soma de todos os bens e serviços finais produzidos em um país durante um período. É o principal indicador para medir o crescimento ou encolhimento da economia.

  - PIB em alta: Economia crescendo.

  - PIB em baixa (especialmente 2 trimestres seguidos): Recessão técnica.

- **Hiato do Produto:** É a diferença entre o PIB efetivo (o que a economia está produzindo de fato) e o PIB potencial (o que a economia poderia produzir se usasse todos os seus recursos de forma eficiente, sem gerar inflação).

  - Hiato positivo: A economia está "superaquecida" (produzindo acima do potencial), o que gera pressão inflacionária.

  - Hiato negativo: A economia está "ociosa" (com desemprego, fábricas paradas), o que tende a reduzir a inflação.
 
## Jargões do Próprio Banco Central

- **Meta de Inflação:** O objetivo central do BACEN. O Conselho Monetário Nacional (CMN) define um valor e um intervalo de tolerância. O BACEN tem que usar a Selic para fazer o IPCA ficar dentro dessa faixa.

- **Balanço de Riscos:** É a avaliação do Copom sobre o que pode fazer a inflação subir (riscos altistas) ou cair (riscos baixistas) no futuro.

  - Exemplo Risco Altista: Uma seca que aumenta o preço dos alimentos ou uma alta do dólar.

  - Exemplo Risco Baixista: Uma recessão global que diminui o preço das commodities.

- **Projeções** (Cenário de Referência): O BACEN cria modelos para prever como a inflação e o PIB vão se comportar nos próximos trimestres e anos. Os relatórios são cheios dessas projeções, que são o que o nosso Bot vai adorar encontrar.

- **Forward Guidance:** É a comunicação do BACEN sobre o que ele pretende fazer com os juros nas próximas reuniões. Isso serve para "guiar" as expectativas do mercado.
