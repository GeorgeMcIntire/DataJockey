import chromadb

def initialize_client(persist_dir):
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    return chroma_client