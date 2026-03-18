import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

pdf_folder = "./pdfs"

all_chunks = []

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        print(f"Loading: {filename}")
        
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)

if not all_chunks:
    print("No valid content found.")
    exit()

embeddings = HuggingFaceEmbeddings()

db = Chroma.from_documents(
    all_chunks,
    embeddings,
    persist_directory="./chroma_db"
)

db.persist()

print("✅ Vector store created successfully!")