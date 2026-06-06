from document_loader import load_documents
from text_splitter import split_documents
from vector_store import (
    build_faiss_index,
    search_documents
)

print("Loading documents...")

documents = load_documents(
    "documents"
)

print(
    f"Documents loaded: {len(documents)}"
)

chunks = split_documents(
    documents
)

print(
    f"Chunks created: {len(chunks)}"
)

build_faiss_index(
    chunks
)

print("FAISS index created.")

results = search_documents(
    "company leave policy"
)

print("\nSearch Results\n")

for doc in results:

    print("-" * 50)

    print(doc.page_content[:500])