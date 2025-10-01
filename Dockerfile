# Multi-stage Dockerfile for Railway deployment with Ollama and Llama 3.2:1b
FROM ollama/ollama:latest AS ollama-base

# Install curl, git, and other dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Application setup
FROM ollama-base

# Create app directory
WORKDIR /app

# Create virtual environment
RUN python3 -m venv /opt/venv

# Activate virtual environment and install dependencies
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# PRE-DOWNLOAD MODEL DURING BUILD
RUN ollama serve & \
    OLLAMA_PID=$! && \
    echo "Waiting for Ollama to start..." && \
    sleep 10 && \
    echo "Downloading model during build..." && \
    ollama pull llama3.2:1b && \
    echo "Model downloaded successfully!" && \
    kill $OLLAMA_PID

# Create a simplified startup script
RUN cat > start.sh <<'EOF' && chmod +x start.sh
#!/bin/bash

# Function to check if Ollama is running
check_ollama() {
    curl -f http://localhost:11434/api/tags > /dev/null 2>&1
}

# Start Ollama server in background
echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready with timeout
echo "Waiting for Ollama to start..."
TIMEOUT=60
COUNTER=0

while ! check_ollama; do
    if [ $COUNTER -ge $TIMEOUT ]; then
        echo "Timeout waiting for Ollama to start"
        exit 1
    fi
    echo "Waiting for Ollama... ($COUNTER/$TIMEOUT seconds)"
    sleep 2
    COUNTER=$((COUNTER+2))
done

echo "Ollama is ready!"

# Verify model is available
echo "Available models:"
ollama list

# Set OLLAMA_HOST for Django app
export OLLAMA_HOST=localhost:11434

# Keep Ollama running and start your Django application
echo "Starting Django application on port ${PORT:-8000}..."
exec python3 manage.py runserver 0.0.0.0:${PORT:-8000}
EOF

# Set environment variables
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_ORIGINS="*"
ENV PORT=8000

# Expose port for the web service
EXPOSE 8000

# Start both Ollama and your application
CMD ["/bin/bash", "/app/start.sh"]