FROM python:3.11

WORKDIR /app

# Copy project files
COPY . .

# Install dependencies (upgraded pip for better compatibility)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Build knowledge base on image creation
RUN python -m main build-kb

# Allow CLI commands to be passed as arguments
ENTRYPOINT ["python", "-m", "main"]
CMD ["--help"]