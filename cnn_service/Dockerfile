FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for image processing
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code
COPY main.py .

# Copy the entire model folder (saved_model.pb + variables/)
# This creates /app/my_model/ inside the container
COPY my_model/ ./my_model/

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the API using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
