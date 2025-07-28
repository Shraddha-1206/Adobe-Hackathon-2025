FROM python:3.10-slim

WORKDIR /app

# Copy project files and folders
COPY run.py .
COPY scripts/ scripts/

# Install dependencies
RUN pip install --no-cache-dir \
    PyPDF2 \
    sentence-transformers \
    transformers \
    scikit-learn \
    joblib

CMD ["python", "run.py"]
