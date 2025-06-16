import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

# Load env variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 1. Load documents and split
loader = PyPDFLoader("documents/mental_health.pdf")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)

# 2. Create vector store using embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
vectorstore = Chroma.from_documents(docs, embedding_model, persist_directory="chroma_db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Gemini Flash LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)

# 4. Prompt template (optional customization)
prompt_template = PromptTemplate(
    input_variables=["chat_history", "question", "context"],
    template="""
You are CalmBot, a kind and supportive mental health assistant.
Answer the user using the context below. Be empathetic, helpful, and never give medical advice.

Context:
{context}

Conversation History:
{chat_history}

User Question:
{question}

CalmBot:"""
)

# 5. Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 6. RAG Chain
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    verbose=False,
)

# 7. Streamlit UI
st.set_page_config(page_title="CalmBot RAG Chat", page_icon="💬")
st.title("🧘 CalmBot - Mental Health Support (RAG + Gemini Flash)")

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

user_input = st.text_input("You:", placeholder="What's on your mind today?")

if user_input:
    result = rag_chain({"question": user_input})
    answer = result["answer"]

    st.session_state.chat_log.append(("You", user_input))
    st.session_state.chat_log.append(("CalmBot", answer))

    # Save to CSV
    log_df = pd.DataFrame(st.session_state.chat_log, columns=["Sender", "Message"])
    log_df.to_csv("chat_log.csv", index=False)

# Show chat history
for sender, msg in st.session_state.chat_log:
    st.markdown(f"**{sender}:** {msg}")
