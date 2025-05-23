import argparse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Please provide a detailed, comprehensive response to the following question. 
Answer in the first person perspective, covering:
- A direct answer upfront
- Supporting details and reasoning in breif, dont make it long
- Add one line TLDR; if response is very long, only if text response exeedes 1000 characters

Question: {question}

Important: If your response is getting long, make sure to properly conclude it rather than cutting off abruptly.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Load embedding model
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Connect to Chroma DB
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Perform similarity search (no relevance scores to avoid the issue)
    results = db.similarity_search(query_text, k=3)

    if len(results) == 0:
        print("Unable to find matching results. Please contact me at prateeksavanur@duck.com. Sorry for the inconvenience.")
        return

    # Prepare context from retrieved documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])

    # Create prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(f"\n[Prompt Sent to Model]\n{prompt}\n")

    # Query the model
    model = ChatGroq(
        model_name="llama3-8b-8192",
        max_tokens=256,  
        temperature=0.4 ,
        top_p=0.9
    )
    response_text = model.predict(prompt)

    # Collect sources metadata
    sources = [doc.metadata.get("source", "No source") for doc in results]
    formatted_response = f"Response:\n{response_text}\n\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()
