# Multi-stage Dockerfile for Railway deployment with Ollama and Llama 3.2:1b
FROM ollama/ollama:latest AS ollama-base

# Install curl and other dependencies
RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Application setup
FROM ollama-base

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a startup script that handles Ollama setup and model download
RUN cat > start.sh << 'EOF'
#!/bin/bash

# Start Ollama server in background
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
sleep 10

# Check if Ollama is responding
for i in {1..30}; do
    if curl -f http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready!"
        break
    fi
    echo "Waiting for Ollama... attempt $i/30"
    sleep 2
done

# Pull the Llama 3.2:1b model
echo "Downloading Llama 3.2:1b model..."
ollama pull llama3.2:1b

# Verify model is available
echo "Available models:"
ollama list

# Keep Ollama running and start your Django/Flask application
echo "Starting main application..."
exec python3 manage.py runserver 0.0.0.0:$PORT
EOF

RUN chmod +x start.sh

# Expose port for the web service
EXPOSE $PORT

# Set environment variables
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_ORIGINS="*"

# Start both Ollama and your application
CMD ["./start.sh"]