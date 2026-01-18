from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.chat_models import ChatLiteLLM
from langchain.chains.retrieval import create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
# -----------------------------
# 1. Create document
# -----------------------------
doc_text = """
Elon Musk is a technology entrepreneur and engineer known for founding SpaceX and Tesla.
He was born on June 28, 1971, in Pretoria, South Africa.
His major achievements include advancing space exploration and electric vehicles.
Musk is also involved with Neuralink and The Boring Company.
This document provides a brief overview of Musk's background and accomplishments.
"""

document = Document(
    page_content=doc_text,
    metadata={"source": "in-memory-doc"}
)

# -----------------------------
# 2. Split text into chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs_split = text_splitter.split_documents([document])

# -----------------------------
# 3. Create embeddings
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# 4. Store in ChromaDB
# -----------------------------
vectorstore = Chroma.from_documents(
    documents=docs_split,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

vectorstore.persist()

# -----------------------------
# 5. Initialize LLM (LiteLLM with fallback)
# -----------------------------
llm = ChatLiteLLM(
    model="gemini/gemini-2.5-flash",
    temperature=0.0,
    max_tokens=None,
    fallbacks=[
        "groq/llama3-8b-8192",
        "groq/mixtral-8x7b-32768"
    ]
)

# -----------------------------
# 6. Build Retrieval Chain
# -----------------------------
retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_template(
    """Answer the question using the given context.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{input}
"""
)

combine_docs_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)

qa_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=combine_docs_chain
)

# -----------------------------
# 7. Query
# -----------------------------
query = "Who is Elon Musk and what are his major achievements?"

result = qa_chain.invoke({"input": query})

print("Answer:", result["answer"])

print("\nSource Documents:")
for doc in result["context"]:
    print("-", doc.page_content[:300])

           
#------------------------------
# load_dotenv()
# client = genai.Client()

# result = client.models.embed_content(
#         model="gemini-embedding-001",
#         contents="What is the meaning of life?"
# )

# print(result.embeddings)
#-----------------------------------