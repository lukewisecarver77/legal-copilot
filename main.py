from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import anthropic
import os
import math

app = FastAPI()

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

app.mount("/static", StaticFiles(directory="static"), name="static")

document_store = {}



# Primary function logic

def chunk_text(text, chunk_size=400, overlap=80):
    """Split text into overlapping chunks"""
    words = text.strip().split()
    chunks = []
    step = chunk_size - overlap
    
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append({"index": len(chunks), "text": chunk})
        
    return chunks

def score_chunk(query, chunk):
    """Score chunks of document text against the query"""
    query_terms = set(query.lower().split())
    chunk_words = chunk["text"].lower().split()
    overlap = len(set(chunk_words) & query_terms)
    return overlap / math.sqrt(len(chunk_words))

def retrieve_chunks(query, chunks, top_k=4):
    """Return top scored chunks"""
    def get_score(c):
        return score_chunk(query, c)
    scored = sorted(chunks, key=get_score, reverse=True)
    return scored[:top_k]




# Request classes

class UploadRequest(BaseModel):
    session_id: str
    text: str
    
class QueryRequest(BaseModel):
    session_id: str
    question: str
    




# Application Routes

@app.get("/")
async def root():
    return FileResponse("static/index.html")



@app.post("/api/upload")
async def upload(req: UploadRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="Document text is empty.")
    chunks = chunk_text(req.text)
    document_store[req.session_id] = chunks
    return {"chunk_count": len(chunks), "word_count": len(req.text.split())}



@app.post("/api/query")
async def query(req: QueryRequest):
    chunks = document_store.get(req.session_id)
    if not chunks:
        raise HTTPException(status_code=404, detail="No document found. Please upload one first.")
    relevant_chunks = retrieve_chunks(req.question, chunks)
    context_parts = []
    for i, chunk in enumerate(relevant_chunks):
        context_parts.append(f"[EXCERPT {i+1}]\n{chunk['text']}")
    context = "\n\n".join(context_parts)
    
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system="Answer questions based only on the provided document excerpts. If the answer isn't in the excerpts, say so. Be concise and use plain English.",
        messages=[
            {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {req.question}"}
        ]
    )
    answer = message.content[0].text
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))