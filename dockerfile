# Use the official Python 3.10 image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency file first and install requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all remaining app files
COPY . .

# Expose the port expected by Hugging Face Spaces
EXPOSE 7860

# Run ChromaDB initialization before app starts
# If chroma DB already exists, skip re-creating
# RUN python create_database.py || echo "Database already exists or creation failed"

# Start the FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
