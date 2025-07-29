# -------- Stage 1: Build dependencies --------
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (leverage caching)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt --prefix=/install

# -------- Stage 2: Final lightweight image --------
FROM python:3.10-slim

WORKDIR /app

# Install system libraries needed for runtime (not build tools)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy your source code
COPY . .

# Default command
CMD ["python", "gnn2.py"]
