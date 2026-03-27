import faiss
index = faiss.read_index("vector_service/nova_historical.index")
print(f"Your Index Dimension is: {index.d}")
