import json
import faiss
import numpy as np
import os

# Path to your GCP download
GCP_JSON_PATH = "vector_service/data/vector_search_index_data.json"

# Output paths for Docker
INDEX_FILE = "nova_historical.index"
MAPPING_FILE = "id_mapping.json"

embeddings = []
ids = []

print(f"Reading {GCP_JSON_PATH}...")
with open(GCP_JSON_PATH, 'r') as f:
    for line in f:
        item = json.loads(line)
        embeddings.append(item['embedding'])
        ids.append(item['id'])

# Convert to FAISS format
embeddings_np = np.array(embeddings).astype('float32')
dimension = embeddings_np.shape[1]

# Build the index
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_np)

# Save the two files Docker needs
faiss.write_index(index, INDEX_FILE)
with open(MAPPING_FILE, "w") as f:
    json.dump(ids, f)

print(f"✅ Created {INDEX_FILE}")
print(f"✅ Created {MAPPING_FILE}")
