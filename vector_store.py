from langchain_community.vectorstores import FAISS

from embeddings import create_embeddings


def build_faiss_index(chunks):

    embeddings = create_embeddings()

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(
        "faiss_index"
    )

    return vectorstore


def load_faiss_index():

    embeddings = create_embeddings()

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def search_documents(
    query,
    k=3
):

    vectorstore = load_faiss_index()

    results = vectorstore.similarity_search(
        query,
        k=k
    )

    return results