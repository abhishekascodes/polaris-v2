# Base image
FROM python:3.11-slim

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Fix permissions
RUN chown -R user:user /app

# Switch to non-root user
USER user

# Set Python path
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PORT=7860

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

EXPOSE 7860

# Run the dashboard server (has WebSocket + REST + landing page)
CMD ["python", "-m", "uvicorn", "dashboard_server:app", "--host", "0.0.0.0", "--port", "7860"]
