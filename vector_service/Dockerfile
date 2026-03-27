FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn faiss-cpu numpy pydantic

# Copy the service code and the index files you just generated
COPY main.py .
COPY nova_historical.index .
COPY id_mapping.json .

EXPOSE 8001

#CMD ["uvicorn", "main.py", "--host", "0.0.0.0", "--port", "8001"]
# Change "main.py" to "main:app"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
