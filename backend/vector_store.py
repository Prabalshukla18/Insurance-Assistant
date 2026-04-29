from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from load_documents import load_documents

# Step 1: Load LangChain Documents
docs = load_documents()

# Step 2: Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(docs)
print(f"Total Chunks: {len(chunks)}")

# Step 3: Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Create Vector DB
vector_db = FAISS.from_documents(chunks, embeddings)
vector_db.save_local("insurance_faiss_db")

print("Vector database created and saved.")