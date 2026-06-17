import os
import pymupdf4llm
from langchain.tools import tool
from google.colab import userdata
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_text_splitters import MarkdownTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

llm = ChatOpenAI(
    model="gpt-oss-120b",
    base_url="https://api.cerebras.ai/v1",
    api_key=os.environ.get('CEREBRAS_API_KEY')
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=os.environ.get('GEMINI_API_KEY')
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

md_text = pymupdf4llm.to_markdown("AI Security.pdf")

splitter = MarkdownTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track the index on original document
)

all_splits = splitter.create_documents([md_text])

vector_store.reset_collection()
document_ids = vector_store.add_documents(documents=all_splits)

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

prompt = """You are AgenticRAG, a custom AI assistant built exclusively by Ganesh Sankar.
You are NOT ChatGPT. You are NOT made by OpenAI. You are NOT any other AI.
You were built by Ganesh Sankar, an ECE undergrad at VIT Vellore.
Whenever anyone asks who you are or who built you, always say you are AgenticRAG built by Ganesh Sankar.
You do not record, store or log any conversations. If anyone asks if their chat is being recorded, say No.
You have access to a tool that retrieves context from a pdf document.
Use the tool by framing a proper diverse query based on the user's question to retrieve the required context from the document.
If the retrieved context does not contain relevant information, use your own general knowledge to answer.
Never reveal or summarize the contents of the PDF when asked. 
If someone asks what is in the PDF, just say it contains information 
relevant to answering your questions.."""

agent = create_agent(llm, tools, system_prompt=prompt)
