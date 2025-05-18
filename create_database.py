from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import shutil

load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

def main():
    print("Generating data store...")
    generate_data_store()

def generate_data_store():
    print("Loading documents...")
    documents = load_documents()
    if not documents:
        print("No documents loaded. Please check the file paths and encoding.")
        return
    print("splitting text...")
    chunks = split_text(documents)
    if not chunks:
        print("No chunks created. Please check if the documents contain valid content.")
        return
    print("Saving chunks to ChromaDB...")
    save_to_chroma(chunks)

def load_documents():
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".md"):  # Ensuring only .md files are loaded
            path = os.path.join(DATA_PATH, filename)
            try:
                with open(path, 'rb') as file:
                    content = file.read().decode('utf-8', errors='ignore')  # Handle decoding errors
                documents.append(Document(page_content=content, metadata={'source': path}))
                print(f"Successfully loaded {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Example debug: print content and metadata of one chunk
    if chunks:
        document = chunks[10]  # You can adjust the index based on your documents
        print(document.page_content)
        print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, HuggingFaceInferenceAPIEmbeddings(
            api_key=os.getenv("HF_API_KEY"),
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
