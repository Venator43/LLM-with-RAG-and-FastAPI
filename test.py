from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.chat_models import init_chat_model
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate

import getpass
import os

from langchain.tools import tool

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

os.environ["HF_TOKEN"] = ""

ls_key = ""
model_name = "Llama-3.2-1B-Instruct-bnb-4bit"

if not os.path.isdir(model_name):
    model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"

model = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    provider="hyperbolic",  # set your provider here
    # provider="nebius",
    # provider="together",
)

# model = HuggingFacePipeline.from_model_id(
#     model_id=model_name,
#     task="text-generation",
#     pipeline_kwargs=dict(
#         max_new_tokens=512,
#         do_sample=False,
#         repetition_penalty=1.03,
#     ),
#     model_kwargs=dict(
#         device_map="cuda",
#         trust_remote_code=True,
#     ),
# )
chat_model = ChatHuggingFace(llm=model)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store = InMemoryVectorStore(embeddings)

file_path = "AUTENTIFIKASI  PERDA 9 TAHUN 2019.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

ids = vector_store.add_documents(documents=all_splits)

tools = [retrieve_context]

chat_model = chat_model.bind_tools(tools)

prompt = (
    "You have access to a tool that retrieves context from the given PDF document. "
    "Use the tool to help answer user queries."
)
agent = create_agent(chat_model, tools, system_prompt=prompt)

query = (
    "Apa isi Pasal 13 pada dokumen yang diberikan?"
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
# results = vector_store.similarity_search(
#     "apa isi Pasal 13 pada bab III ketertiban umum"
# )

# print(results[0])