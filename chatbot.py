import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import DirectoryLoader

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Paths
DOCS_DIR = "mental_health_docs"
DB_DIR = "chroma_db"
os.makedirs(DOCS_DIR, exist_ok=True)

# --- STREAMLIT UI ---
st.set_page_config(page_title="CalmBot - Mental Health Assistant", page_icon="🧘")
st.title("🧘 CalmBot - Mental Health Assistant")
st.markdown("This assistant answers based on curated mental health documents.")

# --- LOAD & PROCESS DOCUMENTS ---
def load_documents():
    loader = DirectoryLoader(DOCS_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    return text_splitter.split_documents(docs)

docs = load_documents()

# --- VECTOR STORE ---
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- LLM & PROMPT ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)

prompt_template = PromptTemplate(
    input_variables=["chat_history", "question", "context"],
    template="""
You are CalmBot, a kind and supportive mental health assistant.
Use the context to answer empathetically and supportively.
Avoid medical advice. Stay grounded in the provided knowledge.

Context:
{context}

Conversation History:
{chat_history}

User Question:
{question}

CalmBot:"""
)

# --- MEMORY & RAG CHAIN ---
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
)

# --- CHAT UI ---
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []
if "show_sources" not in st.session_state:
    st.session_state.show_sources = False

user_input = st.text_input("You:", placeholder="What's on your mind today?")
st.session_state.show_sources = st.checkbox("📎 Show Sources", value=st.session_state.show_sources)

if user_input:
    result = rag_chain.invoke({"question": user_input})
    answer = result["answer"]
    sources = result.get("source_documents", [])

    st.session_state.chat_log.append(("You", user_input))
    st.session_state.chat_log.append(("CalmBot", answer))

    if st.session_state.show_sources:
        for i, src in enumerate(sources):
            metadata = src.metadata
            source_info = f"📄 Source {i+1}: {metadata.get('source', 'Unknown Source')}"
            st.session_state.chat_log.append(("Source", source_info))

    log_df = pd.DataFrame(st.session_state.chat_log, columns=["Sender", "Message"])
    log_df.to_csv("chat_log.csv", index=False)

# --- DISPLAY CHAT HISTORY ---
for sender, msg in st.session_state.chat_log:
    st.markdown(f"**{sender}:** {msg}")
