import json
import faiss
import numpy as np

embeddings = []
metadata_ids = []

# Load the data you just downloaded from GCS
with open("vector_service/data/vector_search_index_data.json", "r") as f:
    for line in f:
        item = json.loads(line)
        embeddings.append(item["embedding"])
        metadata_ids.append(item["id"])

# Create the local index
dim = len(embeddings[0])
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings).astype('float32'))

# Save it
faiss.write_index(index, "vector_service/nova_historical.index")
print("✅ Local index is ready for the Vector Service!")
