# Install required packages first:
# pip install langchain langchain-community pypdf sentence-transformers faiss-cpu transformers torch

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

model_name = "Llama-3.2-1B-Instruct-bnb-4bit"
if not os.path.isdir(model_name):
    model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
    
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

# llm = HuggingFacePipeline.from_model_id(
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

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-235B-A22B-Thinking-2507",  # Changed to compatible model
    temperature=0.6,
    max_new_tokens=512,
    huggingfacehub_api_token="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"
)

llm = ChatHuggingFace(llm=llm)
pdf_loader = PyPDFLoader("AUTENTIFIKASI  PERDA 9 TAHUN 2019.pdf")
documents = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
docs_split = text_splitter.split_documents(documents)

vectorstore = FAISS.from_documents(
    docs_split,
    embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

prompt_template = """Use the following context to answer the question."

Context: {context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

query = "Apa isi Pasal 13 bab III pada dokumen yang diberikan? apa saja aturan yang dilarang?"
result = qa_chain({"query": query})

print(f"Question: {query}")
print(f"Answer: {result['result']}")
print(f"\nSource documents: {len(result['source_documents'])}")
