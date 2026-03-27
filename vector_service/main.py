import faiss
import numpy as np
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Aperion Vector Service")

# Load the index and the ID mapping we just created
index = faiss.read_index("nova_historical.index")
with open("id_mapping.json", "r") as f:
    id_map = json.load(f)

class SearchRequest(BaseModel):
    embedding: list  # The vector from your embedding model
    top_k: int = 5

@app.get("/health")
def health():
    return {"status": "online", "items_indexed": index.ntotal}

@app.post("/search")
async def search(request: SearchRequest):
    try:
        query_vector = np.array([request.embedding]).astype('float32')
        distances, indices = index.search(query_vector, request.top_k)
        
        # Map the FAISS indices back to your actual Apparel IDs
        results = [
            {"id": id_map[i], "distance": float(d)} 
            for d, i in zip(distances[0], indices[0]) if i != -1
        ]
        
        return {"matches": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
