# Stage 1: Build Environment
FROM python:3.11-slim AS builder

# Set the working directory
WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt --target=/app/dependencies

# Stage 2: Runtime Environment
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies and application files
COPY --from=builder /app/dependencies /app/dependencies
COPY src/ /app/src/
COPY vector_db/ /app/vector_db/

# Update the Python path
ENV PYTHONPATH=/app/dependencies:$PYTHONPATH

# Expose the application port
EXPOSE 8080

# Set the entry point
CMD ["python", "src/main.py"]
