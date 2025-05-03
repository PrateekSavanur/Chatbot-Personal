from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

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

def get_chat_response(query_text):
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search(query_text, k=3)
    
    if not results:
        return {"error": "No matching results found."}
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(context=context_text, question=query_text)
    
    model = ChatGroq(
        model_name="llama3-8b-8192",
        max_tokens=256,
        temperature=0.4,
        top_p=0.9
    )
    response_text = model.predict(prompt)
    sources = [doc.metadata.get("source", "No source") for doc in results]
    
    return {
        "response": response_text,
        "sources": sources
    }

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        query_text = json.loads(post_data).get('query_text', '')
        
        result = get_chat_response(query_text)
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())