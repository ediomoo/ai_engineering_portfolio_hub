# Use a slim python image for effeciency
FROM python:3.11-slim

# Set environment variables to prevent .pyc files and enable logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONBUFFERED=1

WORKDIR /app

# Insall system dependencies for ML libraries (like XGBoost)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

#Run the Master Hub
CMD ["streamlit", "run", "main_hub.py", "--server.port=8501", "--server.address=0.0.0.0"]