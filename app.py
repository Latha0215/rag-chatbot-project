import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

#  Paste your Groq API key below
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

embeddings = HuggingFaceEmbeddings()

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(),
    memory=memory
)

st.set_page_config(page_title="Multi PDF Chatbot")
st.title("💬 Chat With Your PDFs")

question = st.text_input("Ask something about your documents:")

if question:
    response = qa_chain.run(question)
    st.write("### Answer:")
    st.write(response)