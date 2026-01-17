rag based chatbot 
=>refrence from pdf and docx
=>evaluate faithfullness relavance and hallucstination



--------------------------------------------

API(fast)
->post(inputtext)
->post(file(pdf,docs))

langchain(gemini,embeddings)
vector db(chroma db)


__________________________________________________

User
 │
 │  (question / pdf / docx)
 ▼
FastAPI
 │
 ├── Document Ingestion API
 │       └── Loader (PDF / DOCX)
 │       └── Text Splitter
 │       └── Embeddings (Gemini)
 │       └── Vector Store (ChromaDB)
 │
 └── Query API
         └── Embed query
         └── Similarity Search (ChromaDB)
         └── Context Retrieval
         └── LLM (Gemini)
         └── Response
         └── Evaluation (Faithfulness, Relevance, Hallucination)