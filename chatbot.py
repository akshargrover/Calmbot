import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
#Langsmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

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
# from datetime import datetime

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Paths
DOCS_DIR = "mental_health_docs"
DB_DIR = "chroma_db"
os.makedirs(DOCS_DIR, exist_ok=True)

# --- STREAMLIT UI ---
st.set_page_config(page_title="CalmBot - Mental Health RAG", page_icon="🧘")
st.title("🧘 CalmBot - Mental Health Chatbot")
st.markdown("This assistant answers based on curated mental health documents.")

# --- LOAD & PROCESS DOCUMENTS ---
def load_documents():
    loader = DirectoryLoader(DOCS_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return text_splitter.split_documents(docs)

docs = load_documents()

# --- VECTOR STORE ---
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- LLM & PROMPT ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)

prompt_template = PromptTemplate(
    input_variables=["chat_history", "question", "context"],
    template="""
You are CalmBot, a compassionate mental health assistant.
Use the provided context to answer with empathy, clarity, and support.

If the context does not clearly answer the user's question,
you may rely on your own general understanding to help—
but always avoid giving any medical or diagnostic advice.

Your answer can be a combination of the context and your own general understanding.

Your role is to support, listen, and guide.
You must answer in a calm, friendly, and emotionally supportive tone.

--------------------------
Relevant Context (from documents):
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

# --- Predefined Prompt Buttons ---
st.subheader("Quick Prompts")
col1, col2 = st.columns(2)
with col1:
    if st.button("😟 I feel anxious"):
        st.session_state.user_input = "I'm feeling anxious, what should I do?"
    if st.button("📚 I can't focus on my studies"):
        st.session_state.user_input = "I'm feeling distracted and can't focus on studies. Any tips?"
with col2:
    if st.button("💤 I can't sleep at night"):
        st.session_state.user_input = "I'm having trouble sleeping. What can help me fall asleep?"
    if st.button("😞 I feel sad"):
        st.session_state.user_input = "I'm feeling very low today. What should I do?"

# --- User input ---
user_input = st.text_input("You:", placeholder="What's on your mind today?")
if user_input:
    st.session_state.user_input = user_input

st.session_state.show_sources = st.checkbox("📎 Show Sources", value=st.session_state.show_sources)

# --- Response generation ---
if "user_input" in st.session_state and st.session_state.user_input:
    query = st.session_state.user_input
    result = rag_chain.invoke({"question": query})
    answer = result["answer"]
    sources = result.get("source_documents", [])

    st.session_state.chat_log.insert(0, ("You", query))
    st.session_state.chat_log.insert(0, ("CalmBot", answer))

    if st.session_state.show_sources:
        for i, src in enumerate(sources):
            metadata = src.metadata
            source_info = f"📄 Source {i+1}: {metadata.get('source', 'Unknown Source')}"
            st.session_state.chat_log.insert(0, ("Source", source_info))

    log_df = pd.DataFrame(st.session_state.chat_log, columns=["Sender", "Message"])
    log_df.to_csv("chat_log.csv", index=False)
    st.session_state.user_input = ""

# --- DISPLAY CHAT HISTORY (Newest First) ---
for sender, msg in st.session_state.chat_log:
    st.markdown(f"**{sender}:** {msg}")