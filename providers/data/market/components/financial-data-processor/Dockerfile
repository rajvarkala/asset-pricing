FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire components directory to get db-interface
COPY . /app

# Install financial-data-processor with dependencies
RUN pip install --no-cache-dir -e /app

ENTRYPOINT ["financial-data-processor"]
