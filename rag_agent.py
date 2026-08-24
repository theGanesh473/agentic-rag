import os
import streamlit as st
import pymupdf4llm
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import MarkdownTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

os.environ["LANGSMITH_TRACING"] = st.secrets["LANGSMITH_TRACING"]
os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=st.secrets["GEMINI_API_KEY"]
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=st.secrets["GEMINI_API_KEY"]
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

vector_store.reset_collection()
md_text = pymupdf4llm.to_markdown("AI Security.pdf")
splitter = MarkdownTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = splitter.create_documents([md_text])
document_ids = vector_store.add_documents(documents=all_splits)
print(f"✅ Embedded {len(all_splits)} chunks into vector store")

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve_context]

import os
import streamlit as st
import pymupdf4llm
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import MarkdownTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

os.environ["LANGSMITH_TRACING"] = st.secrets["LANGSMITH_TRACING"]
os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=st.secrets["GEMINI_API_KEY"]
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=st.secrets["GEMINI_API_KEY"]
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

vector_store.reset_collection()
md_text = pymupdf4llm.to_markdown("AI Security.pdf")
splitter = MarkdownTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = splitter.create_documents([md_text])
document_ids = vector_store.add_documents(documents=all_splits)
print(f"✅ Embedded {len(all_splits)} chunks into vector store")

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve_context]

prompt = """You are a compassionate and non-judgmental mental health support companion for college students.

Your role is to:
- Listen actively and make the student feel heard and validated
- Respond with empathy, warmth and patience
- Help students identify and articulate what they are feeling
- Suggest healthy coping strategies when appropriate
- Gently encourage professional help without being dismissive of their feelings

You must NEVER:
- Diagnose any mental health condition
- Replace or imitate a licensed therapist or counselor
- Dismiss or minimise what the student is feeling
- Give advice that could cause harm

If a student expresses any thoughts of self-harm, suicide, or severe distress:
- Acknowledge their feelings with empathy first
- Gently but clearly encourage them to reach out to a counselor immediately
- Provide these resources:
  * iCall (India): 9152987821
  * Vandrevala Foundation: 1860-2662-345 (24/7)
  * Campus counselor: [add your college counselor contact here]

Always remind the student they are not alone and that help is available.
You are a first point of contact, not a replacement for real professional support."""
agent = create_agent(llm, tools, system_prompt=prompt)


agent = create_agent(llm, tools, system_prompt=prompt)
