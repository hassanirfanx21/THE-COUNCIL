FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (Docker caching trick)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Hugging Face Spaces ALWAYS needs port 7860
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
