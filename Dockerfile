FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies & uv
RUN apt-get update && apt-get install -y curl && \
    curl -Ls https://astral.sh/uv/install.sh | bash

# Add uv to PATH
ENV PATH="/root/.cargo/bin:$PATH"

# Copy dependency file
COPY requirements.txt .

# Install dependencies using uv
RUN uv pip install -r requirements.txt

# Copy all source files
COPY . .

# Expose the port Cloud Run expects
EXPOSE 8080

# Run Flask app
CMD ["flask", "--app", "app/flask_api", "run", "--host=0.0.0.0", "--port=8080"]