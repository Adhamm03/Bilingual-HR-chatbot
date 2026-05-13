FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for FAISS)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application code
COPY . .

# Expose the API port
EXPOSE 8080

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-large')"

# Start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]