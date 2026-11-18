from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# ============================================================
# FASE 1: PREPARAR (roda uma vez para montar o banco)
# ============================================================

# Passo 1 - Document loader: lê o PDF e extrai o texto
caminho_pdf = "databricks_ai_sre_guia.pdf"
loader = PyPDFLoader(caminho_pdf)
paginas = loader.load()

# Junta o texto de todas as páginas em uma única variável
texto_completo = "\n".join([pagina.page_content for pagina in paginas])

# Passo 2 - Text splitter: corta o texto em pedaços (chunks)
# chunk_size manda no tamanho do corte; chunk_overlap é a "costura" entre pedaços
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.create_documents([texto_completo])

# Pega só o texto de cada chunk (fica em .page_content)
textos = [doc.page_content for doc in texts]

# Passo 3 - Embedding model: cada chunk vira um vetor de 1024 números
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
vetores = model.encode_document(textos)
print(vetores.shape)  # mostra (quantidade de chunks, 1024)

# Passo 4 - Vector store: guarda textos e vetores no Chroma (salvo em disco)
# Se mudar o PDF ou o chunk_size, apague a pasta banco_chroma antes de rodar
cliente = chromadb.PersistentClient(path="banco_chroma")
colecao = cliente.get_or_create_collection(name="guia_databricks")

# upsert: pode rodar várias vezes sem dar erro de id repetido
colecao.upsert(
    ids=[str(i) for i in range(len(textos))],
    documents=textos,
    embeddings=vetores.tolist(),
)

# ============================================================
# FASE 2: PERGUNTAR (roda a cada pergunta)
# ============================================================

# Passo 5 - Retriever: a pergunta vira vetor e o Chroma devolve
# os 5 chunks mais parecidos
pergunta = input("Faça aqui sua pergunta: ")
vetor_pergunta = model.encode_query(pergunta)

resultado = colecao.query(
    query_embeddings=[vetor_pergunta.tolist()],
    n_results=5,
)

# Junta os 5 chunks em um único texto, que vai para o prompt
contexto = "\n\n".join(resultado["documents"][0])

# Passo 6 - Chat model: o Qwen (no Ollama) responde usando só o contexto
# Certifique-se de que o modelo já foi baixado (ollama pull qwen3.5:9b)
llm = ChatOllama(model="qwen3.5:9b", temperature=0)

# Prompt: system (as regras) + human (o contexto e a pergunta)
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "Responda usando apenas as informações do documento fornecido."
            " Me de no maximo 7 linhas de resposta."
            " Se a resposta não estiver no documento, diga que não sabe."
        ),
    ),
    ("human", "Documento:\n{contexto}\n\nPergunta: {pergunta}"),
])

# Cadeia (chain): prompt -> modelo
chain = prompt | llm

resposta = chain.invoke({
    "contexto": contexto,
    "pergunta": pergunta,
})

print(resposta.content)