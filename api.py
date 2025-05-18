from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import os
from create_database import main as create_chroma_db

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Please provide a detailed, comprehensive response to the following question. 
Answer in the first person perspective, covering:
- A direct answer upfront
- Supporting details and reasoning in brief
- Add one line TLDR; if response is very long (only if text response exceeds 1000 characters)

Question: {question}

Important: If your response is getting long, make sure to properly conclude it.
"""

groq_api_key = os.getenv("GROQ_API_KEY")

embedding_function = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HF_API_KEY"),
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
model = ChatGroq(
    api_key=groq_api_key,
    model_name="llama3-8b-8192",
    max_tokens=256,
    temperature=0.4,
)

class QueryRequest(BaseModel):
    query_text: str

@app.post("/query")
async def process_query(request: QueryRequest):
    try:
        # Perform similarity search
        results = db.similarity_search(request.query_text, k=3)

        if not results:
            return {
                "response": "Unable to find matching results. Please contact me at prateeksavanur@duck.com.",
                "sources": []
            }

        # Prepare context
        context_text = "\n\n---\n\n".join([doc.page_content for doc in results])

        # Create prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=request.query_text)

        # Query the model
        response_text = model.predict(prompt)

        # Collect sources
        sources = [doc.metadata.get("source", "No source") for doc in results]
        
        return {
            "response": response_text,
            "sources": sources
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    