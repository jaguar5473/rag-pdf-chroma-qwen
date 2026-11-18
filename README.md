# RAG com PDF, Chroma e Qwen local

Projeto de estudo: um RAG (Retrieval-Augmented Generation) em Python que responde perguntas sobre um PDF, rodando tudo localmente.

O PDF usado nos testes foi um guia de estudo sobre o AI SRE da Databricks, mas o código funciona com qualquer PDF de texto.

## O que o projeto faz

Você faz uma pergunta no terminal e o programa:

1. Lê o PDF e extrai o texto (document loader).
2. Corta o texto em pedaços de 1000 caracteres, com 100 de sobreposição (text splitter).
3. Transforma cada pedaço em um vetor com o modelo `Qwen/Qwen3-Embedding-0.6B` (embedding model).
4. Guarda textos e vetores em um banco vetorial Chroma, salvo em disco (vector store).
5. Transforma a pergunta em vetor e busca no Chroma os 5 pedaços mais parecidos (retriever).
6. Envia esses 5 pedaços e a pergunta ao `qwen3.5:9b`, rodando no Ollama, que escreve a resposta (chat model).

O modelo é instruído a responder só com o conteúdo do documento e a dizer que não sabe quando a resposta não estiver lá.

## Tecnologias

- Python
- LangChain (loader, splitter, prompt e integração com o Ollama)
- sentence-transformers (embeddings)
- Chroma (banco vetorial)
- Ollama (modelo de linguagem local)

## Como rodar

### 1. Pré-requisitos

- Python 3.10 ou superior
- [Ollama](https://ollama.com) instalado

### 2. Instalar as dependências

```bash
pip install langchain-community langchain-core langchain-ollama langchain-text-splitters sentence-transformers chromadb pypdf
```

### 3. Baixar o modelo de linguagem

```bash
ollama pull qwen3.5:9b
```

### 4. Colocar o PDF

Coloque o seu PDF na pasta do projeto com o nome `databricks_ai_sre_guia.pdf`, ou troque o valor da variável `caminho_pdf` no `rag.py`.

### 5. Executar

```bash
python rag.py
```

Na primeira execução, o modelo de embedding é baixado do Hugging Face e o banco é criado na pasta `banco_chroma`. Nas próximas, o banco é reaproveitado (o `upsert` evita ids repetidos).

Se você trocar o PDF ou o `chunk_size`, apague a pasta `banco_chroma` antes de rodar de novo.

## Limitações conhecidas

- A busca é por similaridade de significado, então a pergunta funciona melhor quando usa palavras parecidas com as do texto. Na pergunta "Quais são as três trilhas da triagem?" a busca não trouxe a tabela certa. Com "O que verifica a trilha saúde da plataforma, análise do serviço e execução de runbooks?" trouxe.
- O `PyPDFLoader` extrai só texto. Imagens e gráficos do PDF são ignorados, e tabelas perdem o formato.
- O modelo de embedding roda na CPU por padrão, e o Qwen "pensa" antes de responder, então a resposta pode demorar.

## Como testei

Com um RAG, não basta ver a resposta final. Conferi também quais pedaços a busca trouxe e comparei a resposta com o PDF. Foi assim que descobri que um `chunk_size` de 500 cortava uma tabela no meio e que o de 1000 resolvia melhor.

## Próximos passos

- Separar o código em duas partes: indexar o PDF e perguntar.
- Tratar imagens do PDF (OCR ou modelo que descreve imagens).
- Testar outros tamanhos de chunk e comparar as respostas.