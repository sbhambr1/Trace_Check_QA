# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all scripts and data
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY train_dataset.json ./

# Set environment variables (if needed)
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden by Kubernetes job)
CMD ["python", "scripts/openthoughts_sft_training.py"]
