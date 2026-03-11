FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements_backend.txt .
RUN pip install --no-cache-dir -r requirements_backend.txt
RUN pip install --no-cache-dir xgboost shap mlflow sqlalchemy psycopg2-binary

# Copy code
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV DATABASE_URL=postgresql://user:password@db:5432/islamic_bank

# Expose port
EXPOSE 8000

# Start command
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
