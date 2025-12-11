import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
    
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-235B-A22B-Thinking-2507",  # Changed to compatible model
    temperature=0.6,
    max_new_tokens=512,
)

llm = ChatHuggingFace(llm=llm)

app = FastAPI(
    title="Your API Name",
    description="Your API Description",
    version="0.1.0"
)

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["content-disposition"]
) 

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

pdf_loader = PyPDFLoader("AUTENTIFIKASI  PERDA 9 TAHUN 2019.pdf")
documents = pdf_loader.load()
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

@app.post("/api/context/")
async def upload_context(file: UploadFile):
    global qa_chain
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        file_location = f"./{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pdf_loader = PyPDFLoader(file_location)
        documents = pdf_loader.load()
        docs_split = text_splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(
            docs_split,
            embedding_model
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )   

        return JSONResponse(
            status_code=200,
            content={
                "message": "File uploaded successfully",
                "filename": file.filename,
                "location": str(file_location),
                "size": file.size
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    finally:
        file.file.close()

    return True

@app.get("/api/chat/")
async def chat_with_llm(query: str):
    global qa_chain

    result = qa_chain({"query": query})

    print(f"Question: {query}")
    print(f"Answer: {result['result']}")
    print(f"\nSource documents: {len(result['source_documents'])}")

    return {"data": result['result'], "sources": [doc.metadata for doc in result['source_documents']]}


if __name__ ==  "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8002, reload=True)