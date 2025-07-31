import os
import hashlib
import requests
import tempfile
import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, SecretStr
from typing import List, Dict
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from pinecone import Pinecone as PineconeClient

# === Environment Setup ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")
PINECONE_INDEX_NAME = "hackrx-rag-index"

# === FastAPI App & Security ===
app = FastAPI(
    title="Policy Q&A API",
    description="An API to answer questions about policy documents using RAG.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# === Logger Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Pydantic Models ===
class HackRxInput(BaseModel):
    documents: str  # Publicly accessible PDF URL
    questions: List[str]

class HackRxOutput(BaseModel):
    answers: List[str]

# === Language Model & Embeddings Setup ===
embedding_model = None
MODEL_CACHE_PATH = "./embedding_model_cache"

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=MODEL_CACHE_PATH
        )
    return embedding_model

llm = ChatGroq(
    api_key=SecretStr(GROQ_API_KEY) if GROQ_API_KEY else None,
    model="gemma2-9b-it",
    temperature=0.2
)

# === QA Prompt Template ===
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert AI legal assistant specializing in Indian insurance policies.
Your task is to answer the user's question with extreme precision based ONLY on the provided context from the policy document.
If no context is provided, answer the question based on your own knowledge.Dont use \n in output response.
*Instructions:*
1.  *Strictly Contextual:* Your answer MUST be derived exclusively from the text provided in the 'Context' section. Do not use any external knowledge.
2.  *Concise and Formal:* Provide a direct, professional answer. Limit your response to 1-3 clear sentences.
3.  *No Fillers:* Do not include phrases like "According to the policy...", "The context states...", or any disclaimers.
4.  *Focus on Key Details:* Extract specific details such as waiting periods, monetary limits, eligibility criteria, and exclusions when relevant.

*Context:*
{context}

*Question:*
{question}

*Answer:*
"""
)

# === Pinecone Setup ===
if PINECONE_API_KEY and PINECONE_HOST:
    pc = PineconeClient(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_HOST)
    logger.info(f"Successfully connected to Pinecone index at host: {PINECONE_HOST}")
else:
    logger.warning("PINECONE_API_KEY or PINECONE_HOST not found. Pinecone integration is disabled.")
    index = None

# === Helper Functions ===
def get_document_hash(url: str) -> str:
    """Create a unique hash for the document URL to use as a namespace."""
    return hashlib.sha256(url.encode()).hexdigest()

# === API Endpoints ===
@app.get("/", tags=["Health Check"])
def read_root():
    """Health check endpoint to confirm the API is running."""
    return {"status": "ok", "message": "Welcome to the Policy Q&A API!"}

@app.post("/hackrx/run", response_model=HackRxOutput, tags=["Q&A"])
def handle_rag_request(payload: HackRxInput):
    """
    Processes a PDF document to answer a list of questions using a RAG pipeline.
    Requires Bearer token authentication.
    """
    pdf_path = None
    try:
        doc_hash = get_document_hash(payload.documents)
        logger.info(f"Processing request for document hash: {doc_hash}")

        if not index:
            raise HTTPException(status_code=500, detail="Pinecone is not configured.")

        # Check if the namespace is already populated to avoid re-processing
        stats = index.describe_index_stats()
        if doc_hash not in stats.get("namespaces", {}):
            logger.info(f"Document not found in Pinecone. Processing and indexing: {payload.documents}")
            
            response = requests.get(payload.documents)
            response.raise_for_status()
            if not response.headers.get("Content-Type", "").startswith("application/pdf"):
                raise HTTPException(status_code=400, detail="Invalid content type. URL must point to a PDF.")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(response.content)
                pdf_path = tmp_file.name

            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(documents)
            
            logger.info(f"Loaded and split into {len(chunks)} chunks. Creating embeddings...")
            embedding_model = get_embedding_model()
            embeddings = embedding_model.embed_documents([chunk.page_content for chunk in chunks])

            logger.info("Indexing to Pinecone...")
            vectors_to_upsert = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vectors_to_upsert.append(
                    (f"vec{i}", embedding, {"text": chunk.page_content})
                )
            
            # Upsert in batches
            for i in range(0, len(vectors_to_upsert), 100):
                batch = vectors_to_upsert[i:i+100]
                index.upsert(vectors=batch, namespace=doc_hash)

            logger.info("Indexing complete.")
        else:
            logger.info(f"Document already processed. Using existing vectors from namespace: {doc_hash}")

        # === QA Chain Execution ===
        qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=qa_prompt)
        answer_list = []

        for q in payload.questions:
            logger.info(f"Answering question: {q}")
            
            # Create embedding for the question
            embedding_model = get_embedding_model()
            question_embedding = embedding_model.embed_query(q)
            
            # Query Pinecone for relevant documents
            query_response = index.query(
                namespace=doc_hash,
                vector=question_embedding,
                top_k=5,
                include_metadata=True
            )
            
            # Extract text from metadata to create LangChain documents
            from langchain.schema import Document
            
            relevant_docs = []
            if query_response.get("matches"):
                for match in query_response.get("matches"):
                    if match.get("metadata") and "text" in match.get("metadata"):
                        relevant_docs.append(Document(page_content=match.get("metadata")["text"]))

            if not relevant_docs:
                logger.warning(f"No relevant documents found for question: {q}")
                answer_list.append("No relevant information found in the document to answer this question.")
                continue
            
            result = qa_chain.run({"input_documents": relevant_docs, "question": q})
            answer_list.append(result.strip())

        logger.info("All questions processed successfully.")
        return {"answers": answer_list}

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download or access the PDF URL: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
    finally:
        if pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)
            logger.info("Temporary PDF file cleaned up.")
